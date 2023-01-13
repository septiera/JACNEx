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
#  - outputFile (optionnal str): full path to save the pdf
#
# Returns a tupple (validCounts, validSampQC), each variable is created here:
#  - validCounts (np.ndarray[float]): counts for exons covered for all samples
# that passed quality control
#  - validSampQC (np.array[int]): validity status for each sample passed quality
#  control (1: valid, 0: invalid), dim = NbSOIs
def SampsQC(counts, SOIs, outputFile=None):
    #### Fixed parameter:
    # threshold to assess the validity of the sample coverage profile.
    # if the difference between the average density of uncovered exons and
    # the average density of covered exons is less than 20%, the sample is invalid.
    signalThreshold = 0.20

    #### To Fill:
    validSampQC = np.ones(len(SOIs), dtype=np.int)

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
        sampFragCounts = counts[:, sampleIndex]

        # smooth the coverage profile with kernel-density estimate using Gaussian kernels
        # - binEdges (np.ndarray[floats]): FPM range
        # - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
        #   dim= len(binEdges)
        binEdges, densityOnFPMRange = mageCNV.slidingWindow.smoothingCoverageProfile(sampFragCounts)

        # recover the threshold of the minimum density means before an increase
        # - minIndex (int): index from "densityMeans" associated with the first lowest
        # observed mean
        # - minMean (float): first lowest observed mean
        (minIndex, minMean) = mageCNV.slidingWindow.findLocalMin(densityOnFPMRange)

        # recover the threshold of the maximum density means after the minimum
        # density means which is associated with the largest covered exons number.
        # - maxIndex (int): index from "densityMeans" associated with the maximum density
        # mean observed
        # - maxMean (float): maximum density mean
        (maxIndex, maxMean) = findLocalMaxPrivate(densityOnFPMRange, minIndex)

        # graphic representation of coverage profiles.
        # returns a pdf in the output folder
        if outputFile:
            coverageProfilPlotPrivate(SOIs[sampleIndex], binEdges, densityOnFPMRange, minIndex, maxIndex, pdf)

        # sample validity assessment
        if (((maxMean - minMean) / maxMean) < signalThreshold):
            logger.warning("Sample %s doesn't pass quality control.",
                           SOIs[sampleIndex])
            validSampQC[sampleIndex] = 0

            #### fill list control
            listtest.append([SOIs[sampleIndex], minIndex, minMean, maxIndex, maxMean, binEdges[minIndex], 0])

        else:
            exons2RMSamp = np.where(sampFragCounts <= binEdges[minIndex])[0]
            if (len(exons2RM) != 0):
                exons2RM = np.intersect1d(exons2RM, exons2RMSamp)
            else:
                exons2RM = exons2RMSamp

            #### fill list control
            listtest.append([SOIs[sampleIndex], minIndex, minMean, maxIndex, maxMean, binEdges[minIndex], len(exons2RMSamp), len(exons2RM)])

    # close the open pdf
    if outputFile:
        pdf.close()

    # filtering the coverage data to recover valid samples and covered exons
    validCounts = np.delete(counts, np.where(validSampQC == 0)[0], axis=1)
    validCounts = np.delete(validCounts, exons2RM, axis=0)

    logger.info("%s/%s uncovered exons number deleted before clustering for %s/%s valid samples.",
                len(exons2RM), len(counts), len(np.where(validSampQC == 1)[0]), len(SOIs))

    return(validCounts, validSampQC, listtest)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###################################
# findLocalMaxPrivate:
#
# Args:
#  - densityOnFPMRange (np.ndarray[float]): probability density for all bins 
#   in the FPM range
# this arguments is from the slidingWindow.smoothingCoverageProfile function.
#  - minIndex (int): index associated with the first lowest observed density 
#   in np.ndarray "densityOnFPMRange"
# this arguments is from the slidingWindow.findLocalMin function.
#
# Returns a tupple (maxIndex, maxDensity), each variable is created here:
#  - maxIndex (int): index from np.ndarray "densityOnFPMRange" associated with 
#   the maximum density observed occurring after the minimum density
#  - maxDensity (float): maximum density 
def findLocalMaxPrivate(densityOnFPMRange, minIndex):
    maxDensity = np.max(densityOnFPMRange[minIndex:])
    maxIndex = np.where(densityOnFPMRange == maxDensity)[0][0]
    return (maxIndex, maxDensity)

###################################
# coverageProfilPlotPrivate:
# generates a plot per patient
# x-axis: the range of FPM bins (every 0.1 between 0 and 10)
# y-axis: exons densities
# black curve: density data smoothed with kernel-density estimate using Gaussian kernels
# red vertical line: minimum FPM threshold, all uncovered exons are below this threshold
# orange vertical line: maximum FPM, corresponds to the FPM value where the density of
# covered exons is the highest.
#
# Args:
# - sampleName (str): sample exact name
# - binEdges (np.ndarray[floats]): FPM range
# - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
#   dim= len(binEdges)
# - minIndex (int): index associated with the first lowest observed mean
# - maxIndex (int): index associated with the maximum density mean observed
# - pdf (matplotlib object): store plots in a single pdf
#
# Returns a pdf file in the output folder
def coverageProfilPlotPrivate(sampleName, binEdges, densityOnFPMRange, minIndex, maxIndex, pdf):
    # Disable interactive mode
    plt.ioff()

    fig = plt.figure(figsize=(6, 6))
    plt.plot(binEdges, densityOnFPMRange, color='black', label='smoothed densities')

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