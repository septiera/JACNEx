import numpy
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import logging
import os
import datetime

####### JACNEx modules
import callCNVs.likelihoods

# prevent matplotlib and PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
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
        # skip filtered exons
        if Ecodes[thisExon] < 0:
            continue
    
        exonFPMs = FPMsSOIs[thisExon, :]
        fpmMax = numpy.ceil(max(exonFPMs))
        x = numpy.linspace(0, numpy.ceil(fpmMax), nbPoints)

        pdfs = callCNVs.likelihoods.calcLikelihoods(x.reshape(1, nbPoints),
                                                    CN0sigma,
                                                    numpy.array([Ecodes[thisExon]]),
                                                    numpy.array([CN2means[thisExon]]),
                                                    numpy.array([CN2sigmas[thisExon]]),
                                                    isHaploid)

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
    CN1Sigma = CN2Mean / 2
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
    eCodeStr = {1: 'CALLED-WITHOUT-CN1', 0: 'CALLED'}
    limY = (max(pdfs[:, :, 2]) + max(pdfs[:, :, 2])/3)

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
    else:
        pdf_keys = ['CN0', 'CN1', 'CN2', 'CN3']
        pdf_indices = [0, 1, 2, 3]
        if ECodes[thisExon] != 1:
            limY = (max(pdfs[:, :, 1]) + max(pdfs[:, :, 1])/4)

    # plot CN states fits
    for key, idx in zip(pdf_keys, pdf_indices):
        if key in labels and labels[key]:
            matplotlib.pyplot.plot(x,
                                   pdfs[:, :, idx].flatten(),
                                   linewidth=3,
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
                  f"{eCodeStr.get(ECodes[thisExon], 'UNKNOWN')}")
    matplotlib.pyplot.title(title_text)
    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("Density")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlim(0, max(x))
    matplotlib.pyplot.ylim(0, limY)
    pdf.savefig(fig)
    matplotlib.pyplot.close()
