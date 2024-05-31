############################################################################################
# Copyright (C) Nicolas Thierry-Mieg and Amandine Septier, 2021-2024
#
# This file is part of JACNEx, written by Nicolas Thierry-Mieg and Amandine Septier
# (CNRS, France)  {Nicolas.Thierry-Mieg,Amandine.Septier}@univ-grenoble-alpes.fr
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
############################################################################################


import numpy
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import logging
import os
import datetime

####### JACNEx modules
import callCNVs.likelihoods
import countFrags.bed

# prevent matplotlib and PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
####################################################
# checkRegionsToPlot:
# do basic syntactic sanity-check of regionsToPlot, which should be a comma-separated
# list of sampleID:chr:start-end .
# If AOK, return a list of lists [str, str, int, int] holding [sampleID, chrom, start, end];
# else raise exception.
def checkRegionsToPlot(regionsToPlot):
    regions = []
    RTPs = regionsToPlot.split(',')
    for rtp in RTPs:
        rtpList = rtp.split(':')
        if len(rtpList) != 3:
            raise Exception("badly formatted regionToPlot, need 3 ':'-separated fields: " + rtp)
        startEnd = rtpList[2].split('-')
        if len(startEnd) != 2:
            raise Exception("badly formatted regionToPlot, need coords as start-end: " + rtp)
        (start, end) = startEnd
        try:
            start = int(start)
            end = int(end)
            if (start < 0) or (start > end):
                raise Exception()
        except Exception:
            raise Exception("badly formatted regionToPlot, must have 0 <= start <= end: " + rtp)
        regions.append([rtpList[0], rtpList[1], start, end])
    return(regions)


###################
# validate and pre-process each regionsToPlot:
# - does the sampleID exist? In what clusters?
# - does the chrom exist? In auto or gono?
# - are there any exons in the coords?
# If NO to any, log the issue and ignore this regionToPlot;
# if YES to all, populate and return clust2regions:
# key==clusterID, value==Dict with key==sampleID and value==list of exonIndexes
# (in the cluster's exons, auto or gono)
def preprocessRegionsToPlot(regionsToPlot, autosomeExons, gonosomeExons, samp2clusts, clustIsValid):
    clust2regions = {}
    if regionsToPlot == "":
        return(clust2regions)

    autosomeExonNCLs = countFrags.bed.buildExonNCLs(autosomeExons)
    gonosomeExonNCLs = countFrags.bed.buildExonNCLs(gonosomeExons)
    for region in checkRegionsToPlot(regionsToPlot):
        (sampleID, chrom, start, end) = region
        regionStr = sampleID + ':' + chrom + ':' + str(start) + '-' + str(end)
        if sampleID not in samp2clusts:
            logger.warning("ignoring bad regionToPlot %s, sample doesn't exist", regionStr)
            continue
        if chrom in autosomeExonNCLs:
            clustType = 'A_'
            exonNCLs = autosomeExonNCLs
        elif chrom in gonosomeExonNCLs:
            clustType = 'G_'
            exonNCLs = gonosomeExonNCLs
        else:
            logger.warning("ignoring bad regionToPlot %s, chrom doesn't exist", regionStr)
            continue

        clusterID = ""
        for clust in samp2clusts[sampleID]:
            if clust.startswith(clustType):
                clusterID = clust
                break

        if not clustIsValid[clusterID]:
            logger.warning("ignoring regionToPlot %s, sample belongs to invalid cluster %s",
                           regionStr, clusterID)
            continue

        overlappedExons = exonNCLs[chrom].find_overlap(start, end)
        if not overlappedExons:
            logger.warning("ignoring regionToPlot %s, region doesn't overlap any exons", regionStr)
            continue
        if clusterID not in clust2regions:
            clust2regions[clusterID] = {}
        if sampleID not in clust2regions[clusterID]:
            clust2regions[clusterID][sampleID] = []
        for exon in overlappedExons:
            exonIndex = exon[2]
            clust2regions[clusterID][sampleID].append(exonIndex)

    return(clust2regions)


###################################
# This function generates a PDF file with histograms of FPM values and overlays
# model likelihoods for different copy number states (CN0, CN1, CN2, CN3).
# Args:
# - exons (list[str, int, int, str]): exon information.
# - exonsToPlot (dict): Dict with key==exonIndex, value==list of lists[sampleIndex, sampleID] for
#                       which we need to plot the FPMs and CN0-CN3+ models
# - Ecodes (numpy.ndarray[ints]): exons filtering codes.
# - FPMsSOIs (numpy.ndarray[floats]): FPM values for the samples of interest.
# - isHaploid (bool): Whether the sample is haploid.
# - CN0sigma (float): Scale parameter for CN0 distribution.
# - CN2means (numpy.ndarray[floats]): means for CN2 distribution.
# - CN2sigmas (numpy.ndarray[floats]): standard deviations for CN2 distribution.
# - fpmCn0 [float]: FPM value for the uncaptured threshold.
# - clusterID [str]
# - plotFile [str]: File path to save the generated PDF.
# Produces plotFile (pdf format), returns nothing.
def plotExons(exons, exonsToPlot, Ecodes, FPMsSOIs, isHaploid, CN0sigma, CN2means, CN2sigmas, fpmCn0, clusterID, plotDir):
    # hardcoded variables
    nbPoints = 1000  # used to generate the x-axis for the likelihood plots
    minBins = 20
    denom = 4  # used to calculate the number of bins based on the number of samples

    # return immediately if exonsToPlot is empty
    if not exonsToPlot:
        return

    # construct the filename
    current_time = datetime.datetime.now().strftime("%y%m%d_%H-%M-%S")
    numExons = len(exonsToPlot)
    plotFile = os.path.join(plotDir, f'{clusterID}_plotExons_{numExons}_{current_time}.pdf')
    outFile = matplotlib.backends.backend_pdf.PdfPages(plotFile)
    matplotlib.pyplot.ioff()

    # number of bins based on the number of samples.
    bins = int(max(minBins, numpy.ceil(FPMsSOIs.shape[1] / denom)))

    for thisExon in exonsToPlot.keys():
        # skip filtered exons : NOT-CAPTURED and FIT-CN2-FAILED
        if (Ecodes[thisExon] == -1) or (Ecodes[thisExon] == -2):
            continue
    
        exonFPMs = FPMsSOIs[thisExon, :]
        fpmMax = numpy.ceil(max(exonFPMs))
        x = numpy.linspace(0, numpy.ceil(fpmMax), nbPoints)

        pdfs = callCNVs.likelihoods.calcLikelihoods(
            x.reshape(1, nbPoints), CN0sigma, numpy.array([Ecodes[thisExon]]),
            numpy.array([CN2means[thisExon]]), numpy.array([CN2sigmas[thisExon]]),
            isHaploid, True)

        labels = getLabels(isHaploid, CN0sigma, CN2means[thisExon], CN2sigmas[thisExon])

        plotHistogramAndPdfs(exonFPMs, bins, x, pdfs, fpmCn0, exons, thisExon, clusterID,
                             Ecodes, exonsToPlot, labels, isHaploid, outFile)
    outFile.close()

###############################################################################
############################ PRIVATE FUNCTIONS #################################
###############################################################################
# Generate labels for the legend based on whether the sample is haploid
def getLabels(isHaploid, CN0Scale, CN2Mean, CN2Sigma):
    # Hardcoded variables
    CN0Mean = 0
    CN1Mean = CN2Mean / 2
    CN1Sigma = CN2Sigma / 2
    CN3Mu = numpy.log(CN2Mean)
    CN3Sigma = 0.50
    CN3Loc = CN2Mean + 2 * CN2Sigma

    # Initialize labels dictionary
    labels = {
        "CN0": fr'CN0 (halfnorm: $\mu$={CN0Mean}, $\sigma$={CN0Scale:.2f})'
    }

    if isHaploid:
        labels["CN2"] = fr'CN1 (norm: $\mu$={CN2Mean:.2f}, $\sigma$={CN2Sigma:.2f})'
        labels["CN3"] = fr'CN2+ (lognorm: $\mu$={CN3Mu:.2f}, $\sigma$={CN3Sigma:.2f}, $\text{{loc}}$={CN3Loc:.2f})'
    else:
        labels["CN1"] = fr'CN1 (norm: $\mu$={CN1Mean:.2f}, $\sigma$={CN1Sigma:.2f})'
        labels["CN2"] = fr'CN2 (norm: $\mu$={CN2Mean:.2f}, $\sigma$={CN2Sigma:.2f})'
        labels["CN3"] = fr'CN3+ (lognorm: $\mu$={CN3Mu:.2f}, $\sigma$={CN3Sigma:.2f}, $\text{{loc}}$={CN3Loc:.2f})'

    return labels

#####################
# Plot histogram and PDFs for an exon
def plotHistogramAndPdfs(exonFPMs, bins, x, pdfs, fpmCn0, exons, thisExon, clusterId, ECodes, exonsToPlot, labels, isHaploid, pdf):

    colors = ['orange', 'red', 'green', 'purple']
    ECodeSTR = {1: 'CALLED-WITHOUT-CN1', 0: 'CALLED', -3:'CN2-LOW-SUPPORT', -4:'CN0-TOO-CLOSE'}
    limY = (max(pdfs[:, :, 1]) + max(pdfs[:, :, 1])/4)

    fig = matplotlib.pyplot.figure(figsize=(15, 10))
    # plot histogram
    matplotlib.pyplot.hist(exonFPMs,
                           bins=bins,
                           edgecolor='black',
                           label='real data',
                           density=True,
                           color='grey')

    # vertical line for uncaptured threshold
    matplotlib.pyplot.axvline(fpmCn0,
                              color='black',
                              linewidth=3,
                              linestyle='dashed',
                              label=f'uncaptured threshold ({fpmCn0:.2f} FPM)')

    # determine the keys and indices based on haploid status
    if isHaploid:
        pdf_keys = ['CN0', 'CN2', 'CN3']
        pdf_indices = [0, 2, 3]
        limY = (max(pdfs[:, :, 2]) + max(pdfs[:, :, 2])/3)
    else:
        pdf_keys = ['CN0', 'CN1', 'CN2', 'CN3']
        pdf_indices = [0, 1, 2, 3]

    # plot CN states fits
    for key, idx in zip(pdf_keys, pdf_indices):
        if key in labels and labels[key]:
            if (ECodes[thisExon] == 1) and (key == 'CN1'):
                linestyleFIT = 'dashed'
            else:
                linestyleFIT = 'solid'
            matplotlib.pyplot.plot(x,
                                   pdfs[:, :, idx].flatten(),
                                   linewidth=3,
                                   linestyle=linestyleFIT,
                                   color=colors[idx],
                                   label=labels[key])

    # vertical line(s) for target sample(s) FPM(s)
    sampleColors = matplotlib.pyplot.cm.tab20(numpy.linspace(0, 1, len(exonsToPlot[thisExon])))
    for sampleInfo, sampleColor in zip(exonsToPlot[thisExon], sampleColors):
        sampleFpm = exonFPMs[sampleInfo[0]]
        matplotlib.pyplot.axvline(sampleFpm,
                                  color=sampleColor,
                                  linewidth=3,
                                  linestyle='dashed',
                                  label=f'{sampleInfo[1]} ({sampleFpm:.2f} FPM)')

    title_text = (f"{clusterId}\n"
                  f"{exons[thisExon][0]}:{exons[thisExon][1]}-{exons[thisExon][2]} {exons[thisExon][3]}\n"
                  f"{ECodeSTR.get(ECodes[thisExon], 'UNKNOWN')}")
    matplotlib.pyplot.title(title_text)
    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("Density")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlim(0, max(x))
    matplotlib.pyplot.ylim(0, limY)
    pdf.savefig(fig)
    matplotlib.pyplot.close()
