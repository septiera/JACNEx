import numpy as np
import logging
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

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
#  - counts (np.ndarray[float]): normalised fragment counts
#  - SOIs (list[str]): samples of interest names
#  - QCPDF (str): full path to save the pdf
#
# Returns a tupple (sampsQCfailed, uncoveredExons), each variable is created here:
#  - sampsQCfailed (list[int]): sample indexes not validated by quality control
#  - uncapturedExons (list[int]): uncaptured exons indexes common to all samples
# passing quality control

def SampsQC(counts, SOIs, QCPDF):
    #### Fixed parameter:
    # threshold to assess the validity of the sample coverage profile.
    signalThreshold = 0.20

    #### To Fill:
    sampsQCfailed = []
    uncoveredExons = []

    # create a matplotlib object and open a pdf
    PDF = matplotlib.backends.backend_pdf.PdfPages(QCPDF)

    for sampleIndex in range(len(SOIs)):
        # extract sample counts
        sampFragCounts = counts[:, sampleIndex]

        # smooth the coverage profile with kernel-density estimate using Gaussian kernels
        # - FPMRange (np.ndarray[floats]): FPM range, [0-10] each 0.01
        # - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
        #   dim= len(binEdges)
        FPMRange, densityOnFPMRange = clusterSamps.smoothing.smoothingCoverageProfile(sampFragCounts)

        # recover the threshold (FPMRange index) of the minimum density before an increase
        # - minDensity2FPMIndex (int): FPMRange index associated with the first lowest
        # observed density
        # - minDensity (float): first lowest density observed
        (minDensity2FPMIndex, minDensity) = clusterSamps.smoothing.findLocalMin(densityOnFPMRange)

        # recover the threshold of the maximum density means after the minimum
        # density means which is associated with the largest covered exons number.
        # - maxDensity2FPMIndex (int): FPMRange index associated with the maximum density
        # observed
        # - maxDensity (float): maximum density
        (maxDensity2FPMIndex, maxDensity) = findLocalMaxPrivate(densityOnFPMRange, minDensity2FPMIndex)

        # graphic representation of coverage profiles.
        # returns a pdf in the plotDir
        coverageProfilPlotPrivate(SOIs[sampleIndex], FPMRange, densityOnFPMRange, minDensity2FPMIndex, maxDensity2FPMIndex, PDF)

        #############
        # sample validity assessment
        if (((maxDensity - minDensity) / maxDensity) <= signalThreshold):
            sampsQCfailed.append(sampleIndex)
        #############
        # uncovered exons lists comparison
        else:
            uncovExonSamp = np.where(sampFragCounts <= FPMRange[minDensity2FPMIndex])[0]
            if (len(uncoveredExons) != 0):
                uncoveredExons = np.intersect1d(uncoveredExons, uncovExonSamp)
            else:
                uncoveredExons = uncovExonSamp

    # close the open pdf
    PDF.close()

    # returns in stderr the results on the filtered data
    logger.info("%s/%s uncovered exons number deleted before clustering for %s/%s valid samples.",
                len(uncoveredExons), len(counts), (len(SOIs) - len(sampsQCfailed)), len(SOIs))

    return(sampsQCfailed, uncoveredExons)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###################################
# findLocalMaxPrivate:
#
# Args:
#  - densityOnFPMRange (np.ndarray[float]): probability density for all bins
#   in the FPM range
# this arguments is from the smoothing.smoothingCoverageProfile function.
#  - minDensity2FPMIndex (int): index associated with the first lowest observed density
#   in np.ndarray "densityOnFPMRange"
# this arguments is from the slidingWindow.findLocalMin function.
#
# Returns a tupple (maxIndex, maxDensity), each variable is created here:
#  - maxDensity2FPMIndex (int): FPMRange index associated with the maximum density
# observed
# - maxDensity (float): maximum density
def findLocalMaxPrivate(densityOnFPMRange, minDensity2FPMIndex):
    maxDensity = np.max(densityOnFPMRange[minDensity2FPMIndex:])
    maxDensity2FPMIndex = np.where(densityOnFPMRange == maxDensity)[0][0]
    return (maxDensity2FPMIndex, maxDensity)


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
