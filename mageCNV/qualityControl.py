import numpy as np
import logging
import mageCNV.slidingWindow
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# SampsQC :
# evaluates the coverage profile of the samples and identifies the uncovered
# exons for all samples.
#
# Args:
#  - counts (np.ndarray[float]): normalised fragment counts.
#  - SOIs (list[str]): samples of interest names
#  - windowSize (int): number of bins in a window
#  - outputFile (optionnal str): full path to save the pdf
#
# Returns a tupple (validityStatus, exons2RMAllSamps), each variable is created here:
#  - validityStatus (np.array[int]): the validity status for each sample
# (1: valid, 0: invalid), dim = NbSOIs
#  - validCounts (np.ndarray[float]): counts for exons covered for all samples
# that passed quality control
def SampsQC(counts, SOIs, windowSize, outputFile=None):
    #### Fixed parameters:
    # FPM threshold limitation with signal to be analysed
    # exons uncovered and most exons covered
    FPMSignal = 10
    # number of bins to create a sufficiently precise range for the FPM (0.1)
    binsNb = 100
    # threshold to assess the validity of the sample coverage profile.
    # if the difference between the average density of uncovered exons and
    # the average density of covered exons is less than 20%, the sample is invalid.
    signalThreshold = 0.20

    #### To Fill:
    validityStatus = np.ones(len(SOIs), dtype=np.int)

    #### Accumulator:
    # a list[int] storing the indixes of uncovered exons in all valid samples
    exons2RM = []

    #### To remove not use only for dev control
    listtest = []

    # create a matplotlib object and open a pdf if the figure option is
    # true in the main script
    if outputFile:
        pdf = matplotlib.backends.backend_pdf.PdfPages(outputFile)

    for sampleIndex in range(len(SOIs)):
        # extract sample counts
        sampCounts = counts[:, sampleIndex]
        # FPM threshold limitation
        sampCounts = sampCounts[sampCounts <= FPMSignal]

        # FPM range creation
        start = 0
        stop = FPMSignal
        # binEdges (np.ndarray[floats]): values at which each FPM bin starts and ends
        binEdges = np.linspace(start, stop, num=binsNb)
        # limitation of decimal points to avoid float approximations
        binEdges = np.around(binEdges, 1)

        # densities (np.ndarray[floats]):
        # number of elements in the bin/(bin width*total number of elements)
        densities = np.histogram(sampCounts, bins=binEdges, density=True)[0]

        # smooth the coverage profile from a sliding window.
        # densityMeans (list[float]): mean density for each window covered
        densityMeans = mageCNV.slidingWindow.smoothingCoverageProfile(densities, windowSize)

        # recover the threshold of the minimum density means before an increase
        # minIndex (int): index from "densityMeans" associated with the first lowest
        # observed mean
        # minMean (float): first lowest observed mean
        (minIndex, minMean) = mageCNV.slidingWindow.findLocalMin(densityMeans)

        # recover the threshold of the maximum density means after the minimum
        # density means which is associated with the largest covered exons number.
        # maxIndex (int): index from "densityMeans" associated with the maximum density
        # mean observed
        # maxMean (float): maximum density mean
        (maxIndex, maxMean) = findLocalMaxPrivate(densityMeans, minIndex)

        # graphic representation of coverage profiles.
        # returns a pdf in the output folder
        if outputFile:
            coverageProfilPlotPrivate(SOIs[sampleIndex], binEdges, densities, densityMeans, minIndex, maxIndex, pdf)

        #### fill list control
        listtest.append([SOIs[sampleIndex], minIndex, minMean, maxIndex, maxMean])

        # sample validity assessment
        if (((maxMean - minMean) / maxMean) < signalThreshold):
            logger.warning("Sample %s has a coverage profile doesn't distinguish between covered and uncovered exons.",
                           SOIs[sampleIndex])
            validityStatus[sampleIndex] = 0
        else:
            exons2RMSamp = np.where(sampCounts <= binEdges[minIndex])
            if (len(exons2RM) != 0):
                exons2RM = np.intersect1d(exons2RM, exons2RMSamp)
            else:
                exons2RM = exons2RMSamp

    # close the open pdf
    if outputFile:
        pdf.close()

    # filtering the coverage data to recover valid samples and covered exons
    validCounts = np.delete(counts, np.where(validityStatus == 0)[0], axis=1)
    validCounts = np.delete(validCounts, exons2RM, axis=0)

    logger.info("%s/%s uncovered exons number deleted before clustering for %s/%s valid samples.",
                len(exons2RM), len(counts), len(np.where(validityStatus == 1)[0]), len(SOIs))

    return(validityStatus, validCounts, listtest)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###################################
# findLocalMaxPrivate:
# recover the threshold of the maximum density means after the
# minimum density means which is associated with the largest covered exons number.
#
# Args:
#  - minDensityMean (list[float]): mean density for each window covered
# this arguments are from the slidingWindow.smoothingCoverProfile function.
#  - minIndex (int): index associated with the first lowest observed mean
# this argument is from the slidingWindow.findLocalMin function.
#
# Returns a tupple (minIndex, minDensity), each variable is created here:
#  - maxIndex (int): index from "densityMeans" associated with the maximum density
#    mean observed
#  - maxMean (float): maximum density mean
def findLocalMaxPrivate(densityMeans, minIndex):
    maxMean = np.max(densityMeans[minIndex:])
    maxIndex = np.where(densityMeans == maxMean)[0][0]
    return (maxIndex, maxMean)


####################################
# coverageProfilPlotPrivate:
# generates a plot per patient
# x-axis: the range of FPM bins (every 0.1 between 0 and 10)
# y-axis: exons densities
# blue curve: raw density data
# orange curve: density data smoothed by moving average.
# red vertical line: minimum FPM threshold, all uncovered exons are below this threshold
# orange vertical line: maximum FPM, corresponds to the FPM value where the density of
# covered exons is the highest.
#
# Args:
# - sampleName (str): sample exact name
# - binEdges (np.ndarray[floats]): values at which each FPM bin starts and ends
# - densities (np.ndarray[floats]): exons densities for each binEdges
# - densityMeans (list[float]): mean density for each window covered
# - minIndex (int): index associated with the first lowest observed mean
# - maxIndex (int): index associated with the maximum density mean observed
# - pdf (matplotlib object): allows you to store plots in a single pdf
#
# Returns a pdf file in the output folder
def coverageProfilPlotPrivate(sampleName, binEdges, densities, densityMeans, minIndex, maxIndex, pdf):
    # Disable interactive mode
    plt.ioff()

    fig = plt.figure(figsize=(6, 6))
    # binEdges defines a bin by its start and end, as an index in densities corresponds
    # to a bin. So total size of both np.ndarray are not equivalent.
    plt.plot(binEdges[:-1], densities, color='black', label='raw density')
    plt.plot(binEdges[:-1], densityMeans, color='mediumblue', label='smoothed density')

    plt.axvline(binEdges[minIndex], color='crimson', linestyle='dashdot', linewidth=2,
                label="minFPM=" + '{:0.1f}'.format(binEdges[minIndex]))
    plt.axvline(binEdges[maxIndex], color='darkorange', linestyle='dashdot', linewidth=2,
                label="maxFPM=" + '{:0.1f}'.format(binEdges[maxIndex]))

    plt.ylim(0, 0.5)
    plt.ylabel("Exon densities")
    plt.xlabel("Fragments Per Million")
    plt.title(sampleName + " coverage profile")
    plt.legend()

    pdf.savefig(fig)
    plt.close()
