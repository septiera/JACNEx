import os
import logging
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot

####### MAGE-CNV modules
import clusterSamps.smoothing

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

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
    capturedExons = np.zeros(len(counts[:, 0]), dtype=np.bool_)

    # create matplotlib PDF objects
    pdfPass = matplotlib.backends.backend_pdf.PdfPages(plotFilePass)
    pdfFail = matplotlib.backends.backend_pdf.PdfPages(plotFileFail)

    for sampleIndex in range(len(SOIs)):
        sampleCounts = counts[:, sampleIndex]

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
        # vertical dashed lines: coordinates (default to 0 ie no lines) and legends
        (xmin, xmax) = (0, 0)
        (line1legend, line2legend) = ("", "")
        # pdf to use for this sample
        pdf = pdfPass

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
            logger.info("sample %s is bad: %s", SOIs[sampleIndex], str(e))
            sampsQCfailed.append(sampleIndex)
            pdf = pdfFail
            # ymax needed for plotting even without a maxIndex... we don't want a huge
            # value close to y(0) => find the max Y after the first 15% of dataRange
            ymax = max(densities[0][int(len(dataRanges[0]) * 0.15):])
        else:
            (xmin, ymin) = (dataRanges[0][minIndex], densities[0][minIndex])
            (xmax, ymax) = (dataRanges[0][maxIndex], densities[0][maxIndex])
            line1legend = "min FPM = " + '{:0.2f}'.format(xmin)
            line2legend = "max FPM = " + '{:0.2f}'.format(xmax)
            # require at least minLow2high fractional increase from ymin to ymax
            if ((ymax - ymin) / ymin) < minLow2high:
                logger.info("sample %s is bad: min (%.2f,%.2f) and max (%.2f,%.2f) densities too close",
                            SOIs[sampleIndex], xmin, ymin, xmax, ymax)
                sampsQCfailed.append(sampleIndex)
                pdf = pdfFail
            else:
                capturedExons = np.logical_or(capturedExons, sampleCounts > xmin)

        # plot all the densities for sampleIndex in a single plot
        title = SOIs[sampleIndex] + " density of exon FPMs"
        # max range on Y axis for visualization, 3*ymax should be fine
        ylim = 3 * ymax
        plotDensities(title, dataRanges, densities, legends, xmin, xmax, line1legend, line2legend, ylim, pdf)

    # close PDFs
    pdfPass.close()
    pdfFail.close()

    if len(sampsQCfailed) > 0:
        logger.warn("%s/%s samples fail exon-density QC (see QC plots), these samples won't get CNV calls",
                    len(sampsQCfailed), len(SOIs))
    logger.info("%s/%s exons are not covered/captured in any sample and will be ignored",
                len(capturedExons[np.logical_not(capturedExons)]), len(capturedExons))

    return(sampsQCfailed, capturedExons)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###################################
# Plot one or more curves, and optionally two vertical dashed lines, on a
# single figure.
# Each curve is passed as an ndarray of X coordinates (eg dataRanges[2] for the
# third curve), a corresponding ndarray of Y coordinates (densities[2]) of the
# same length, and a legend (legends[2]).
# The vertical dashed lines are drawn at X coordinates line1 and line2, unless
# line1==line2==0.
#
# Args:
# - title: plot's title (string)
# - dataRanges: list of N ndarrays storing X coordinates
# - densities: list of N ndarrays storing the corresponding Y coordinates
# - legends: list of N strings identifying each (dataRange,density) pair
# - line1, line2 (floats): X coordinates of dashed vertical lines to draw
# - line1legend, line2legend (strings): legends for the vertical lines
# - ylim (float): Y max plot limit
# - pdf: matplotlib PDF object where the plot will be saved
#
# Returns a pdf file in the output folder
def plotDensities(title, dataRanges, densities, legends, line1, line2, line1legend, line2legend, ylim, pdf):
    # sanity
    if (len(dataRanges) != len(densities)) or (len(dataRanges) != len(legends)):
        raise Exception('plotDensities bad args, length mismatch')

    # set X max plot limit (both axes start at 0)
    xlim = max(dataRanges[:][-1])

    # Disable interactive mode
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(6, 6))
    for i in range(len(dataRanges)):
        matplotlib.pyplot.plot(dataRanges[i], densities[i], label=legends[i])

    if (line1 != 0) or (line2 != 0):
        matplotlib.pyplot.axvline(line1, color='crimson', linestyle='dashdot', linewidth=1, label=line1legend)
        matplotlib.pyplot.axvline(line2, color='darkorange', linestyle='dashdot', linewidth=1, label=line2legend)

    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("density")
    matplotlib.pyplot.xlim(0, xlim)
    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc='upper right', fontsize='small')

    pdf.savefig(fig)
    matplotlib.pyplot.close()
