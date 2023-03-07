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
# With a typical exome, the density plot of FPM counts over all exons appears as
# a mixture of:
# A) a sharp peak at 0 FPM (eg exons that are not captured by the exome kit)
# B) a vaguely bell-shaped curve with a heavy tail (captured exons).
# MAGe-CNV relies on this mixture (the A profile is used to call homo-deletions), and
# cannot call CNVs for samples whose FPM density profile does not fit this expectation.
# Furthermore, some exons have close-to-zero FPM in every sample (eg they are not
# targeted by the capture kits, or are in hard-to-align genomic regions).
# MAGe-CNV cannot make any CNV calls for such exons, they just slow down and
# possibly cloud up the analyses for other exons.
# This function identifies any atypical "QC-failing" samples, and also any
# "never-captured" exons. In addition, FPM density plots for PASSing and FAILing
# samples are produced.
#
# Args:
# - counts (np.ndarray[float]): normalised fragment counts
# - SOIs (list[str]): sample names
# - plotFilePass: pdf filename (with path) where plots for QC-passing samples are made
# - plotFileFail: same as plotFilePass but for QC-failing samples
# - minLow2high (float): a sample's FPM density must increase by a fraction at least
#     minLowToHigh between it's first local min and it's subsequent max to pass QC
# - testBW: if True each plot includes several different KDE bandwidth algorithms/values,
#     for testing and comparing; otherwise use what seems best in our tests
#
# Return (sampsQCfailed, capturedExons):
#  - sampsQCfailed: list of indexes (in SOIs) of samples that fail QC
#  - capturedExons: np.ndarray of bools, length = number of exons, True iff corresponding exon is
#    "captured" (FPM > first local min, see findFirstLocalMinMax) in at least one QC-passing sample
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

        if pdf == pdfFail:
            # need ymax for plotting even if sample is bad, the max Y after the first 15%
            # of dataRange is a good rule of thumb (not too cropped, not too zoomed out)
            ymax = max(densities[0][int(len(dataRanges[0]) * 0.15):])

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
