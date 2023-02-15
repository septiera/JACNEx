import os
import logging
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot

####### MAGE-CNV modules
import clusterSamps.smoothing

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# SampsQC :
# evaluates the coverage profile of the samples and identifies the uncaptured
# exons for all valid samples.
# This allows to identify the non optimal informations (unvalid samples, uncaptured exons)
# for clustering.
# A sample coverage profile is deduced by computing the exon densities from the FPM values.
# To gain accuracy, this coverage profile is smoothed.
# A good coverage profile differentiates between uncovered and covered exons.
# A first density drop is associated with uncovered and uncaptured exons, a threshold is
# deduced from the first lowest density obtained.
# Then the density increases which is the signal associated with captured exons,
# a threshold is deduced from the highest density obtained.
# If the difference between these two thresholds is less than 20% of the highest
# density threshold, the coverage profile is not processable.
# Invalidation of the sample for the rest of the analyses.
# For validated samples, recovery of uncaptured exon indexes that have a FPM
# value lower than the FPM value associated with the lowest density threshold.
# An intersection of the different lists of uncaptured exons is performed in order
# to find those common to all validated samples.
#
# Args:
# - counts (np.ndarray[float]): normalised fragment counts
# - SOIs (list[str]): sample names
# - plotFilePass: pdf filename (with path) where plots for QC-passing samples are made
# - plotFileFail: same as plotFilePass but for QC-failing samples
# - minLow2high (float): a sample's fractional density increase from min to max must
#     be >= minLow2high to pass QC - default should be OK
# - testBW: if True each plot includes several different KDE bandwidth algorithms/values,
#     for testing and comparing; otherwise use what seems best in our tests
#
# Returns a tuple (sampsQCfailed, uncapturedExons), each variable is created here:
#  - sampsQCfailed (list[int]): sample indexes not validated by quality control
#  - uncapturedExons (list[int]): uncaptured exons indexes common to all samples
# passing quality control

def SampsQC(counts, SOIs, plotFilePass, plotFileFail, minLow2high=0.2, testBW=False):
    if os.path.isfile(plotFilePass) or os.path.isfile(plotFileFail):
        logger.error('SampsQC : plotFile(s) %s and/or %s already exist', plotFilePass, plotFileFail)
        raise Exception("plotFilePass and/or plotFileFail already exist")

    #### To Fill:
    sampsQCfailed = []
    uncapturedExons = np.arange(len(counts[:, 0]))

    # create matplotlib PDF objects
    pdfPass = matplotlib.backends.backend_pdf.PdfPages(plotFilePass)
    pdfFail = matplotlib.backends.backend_pdf.PdfPages(plotFileFail)

    for sampleIndex in range(len(SOIs)):
        sampleCounts = counts[:, sampleIndex]
        sampleOK = True

        # for smoothing and plotting we don't care about large counts (and they mess things up),
        # we will only consider the bottom fracDataForSmoothing fraction of counts, default
        # value should be fine
        fracDataForSmoothing = 0.96
        # corresponding max counts value
        maxData = np.quantile(sampleCounts, fracDataForSmoothing)

        # produce smoothed representation(s) (one if testBW==False, several otherwise)
        dataRanges = []
        densities = []
        legends = []

        if testBW:
            allBWs = ('ISJ', 0.15, 0.20, 'scott', 'silverman')
        else:
            # our current favorite, gasp at the python-required trailing comma...
            allBWs = ('ISJ',)

        for bw in allBWs:
            try:
                (dr, dens, bwValue) = clusterSamps.smoothing.smoothData(sampleCounts, maxData=maxData, bandwidth=bw)
            except Exception as e:
                logger.error('smoothing failed for %s : %s', str(bw), repr(e))
                raise
            dataRanges.append(dr)
            densities.append(dens)
            legend = str(bw)
            if isinstance(bw, str):
                legend += ' => bw={:.2f}'.format(bwValue)
            legends.append(legend)

        # find indexes of first local min density and first subsequent local max density,
        # looking only at our first smoothed representation
        try:
            (minIndex, maxIndex) = clusterSamps.smoothing.findFirstLocalMinMax(densities[0])
        except Exception as e:
            logger.warn("sample %s is bad: %s", SOIs[sampleIndex], str(e))
            sampsQCfailed.append(sampleIndex)
            sampleOK = False
        else:
            (xmin, ymin) = (dataRanges[0][minIndex], densities[0][minIndex])
            (xmax, ymax) = (dataRanges[0][maxIndex], densities[0][maxIndex])
            # require at least minLow2high fractional increase from ymin to ymax
            if ((ymax - ymin) / ymin) < minLow2high:
                logger.warn("sample %s is bad: min (%.2f,%.2f) and max (%.2f,%.2f) densities too close",
                            SOIs[sampleIndex], xmin, ymin, xmax, ymax)
                sampsQCfailed.append(sampleIndex)
                sampleOK = False
            else:
                # restrict list of uncaptured exons
                uncovExonSamp = np.where(sampleCounts <= xmin)[0]
                uncapturedExons = np.intersect1d(uncapturedExons, uncovExonSamp)

        # plot all the densities for sampleIndex in a single plot
        title = SOIs[sampleIndex] + " density of exon FPMs"
        if sampleOK:
            plotDensities(title, dataRanges, densities, legends, xmin, xmax, ymax, pdfPass)
        else:
            # arbitrary ymax needed for plotting, we don't want to use a huge
            # value close to y(0) => ignore the first 15% of dataRange
            ymax = max(densities[0][int(len(dataRanges[0]) * 0.15):])
            plotDensities(title, dataRanges, densities, legends, 0, 0, ymax, pdfFail)

    # close PDFs
    pdfPass.close()
    pdfFail.close()

    logger.info("%s/%s uncaptured exons number deleted before clustering for %s/%s valid samples.",
                len(uncapturedExons), len(counts), (len(SOIs) - len(sampsQCfailed)), len(SOIs))

    return(sampsQCfailed, uncapturedExons)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###################################
# coverageProfilPlotPrivate:
# generates a plot per patient
# x-axis: the range of FPM bins (every 0.1 between 0 and 10)
# y-axis: exons densities
# black curve: density data smoothed with kernel-density estimate
# using Gaussian kernels
# red vertical line: minimum FPM threshold, all uncovered and
# uncaptured exons are below this threshold
# orange vertical line: maximum FPM, corresponds to the FPM
# value where the density of captured exons is the highest.
#
# Args:
# - sampleName (str): sample exact name
# - binEdges (np.ndarray[floats]): FPM range
# - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
#   dim= len(binEdges)
# - minIndex (int): index associated with the first lowest density observed
# - maxIndex (int): index associated with the maximum density observed
# - pdf (matplotlib object): store plots in a single pdf
#
# Returns a pdf file in the output folder
def coverageProfilPlotPrivate(sampleName, binEdges, densityOnFPMRange, minIndex, maxIndex, pdf):
    # Disable interactive mode
    matplotlib.pyplot.ioff()

    fig = matplotlib.pyplot.figure(figsize=(6, 6))
    matplotlib.pyplot.plot(binEdges, densityOnFPMRange, color='black', label='smoothed densities')

    matplotlib.pyplot.axvline(binEdges[minIndex], color='crimson', linestyle='dashdot', linewidth=2,
                              label="minFPM=" + '{:0.1f}'.format(binEdges[minIndex]))
    matplotlib.pyplot.axvline(binEdges[maxIndex], color='darkorange', linestyle='dashdot', linewidth=2,
                              label="maxFPM=" + '{:0.1f}'.format(binEdges[maxIndex]))

    matplotlib.pyplot.ylim(0, 0.5)
    matplotlib.pyplot.ylabel("Exon densities")
    matplotlib.pyplot.xlabel("Fragments Per Million")
    matplotlib.pyplot.title(sampleName + " coverage profile")
    matplotlib.pyplot.legend()

    pdf.savefig(fig)
    matplotlib.pyplot.close()
