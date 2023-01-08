import numpy as np
import logging

import slidingWindow

# set up logger, using inherited config
logger = logging.getLogger(__name__)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# SampsQC :
# evaluates the coverage profile of the samples and identifies the uncovered exons for all samples.
# Args:
#  - counts (np.ndarray[float]): normalised fragment counts. dim = NbExons x NbSOIs
#  - SOIs (list[str]): samples of interest names
#  - windowSize (int): number of bins in a window 
# Returns a tupple (validityStatus, exons2RMAllSamps), each variable is created here:
#  - validityStatus (np.array[int]): the validity status for each sample (1: valid, 0: invalid), dim = NbSOIs
#  - exons2RMAllSamps (list[int]): indexes of uncovered exons common for all samples with a valid status

def SampsQC(counts, SOIs, windowSize):
    # To Fill:
    validityStatus = np.ones(len(SOIs), dtype=np.int)

    # Accumulator:
    exons2RMAllSamps = []
    
    for sampleIndex in range(len(SOIs)):
        # extract sample counts
        sampCounts = counts[:, sampleIndex]
        
        # densities calculation by a histogram with a sufficiently accurate range (e.g: 0.01)
        start = 0
        stop = max(sampCounts)
        binEdges = np.linspace(start, stop, int(stop*100)) # !!! hardcode 
        densities, fpmBins = np.histogram(sampCounts, bins=binEdges, density=True)

        # using a sliding window to extract the window index "smallestIndex" with the first minimum sum 
        # density is observed "smallestSum" and a list 'densitiesSums', with each density sum for each window browsed
        (smallestIndex, smallestSum, densitiesSums) = slidingWindow.selectMinThreshold(densities, windowSize)

        # Find the maximum density sum and its window index seen afterwards "smallestSum"
        # don't consider "smallestIndex" as it's corrected according to the "windowSize".
        biggestSum = max(densitiesSums[densitiesSums.index(smallestSum)+1:])
        biggestIndex = densitiesSums.index(biggestSum) + windowSize//2

        # sample removed where the minimum sum (corresponding to the limit of the poorly covered exons) 
        # is less than 15% different from the maximum sum (corresponding to the profile of the most covered exons). 
        if (((biggestSum-smallestSum) / biggestSum) < 0.15):
            logger.warning("Sample %s has a coverage profile doesn't distinguish between covered and \
                uncovered exons", SOIs[sampleIndex])
            validityStatus[sampleIndex] = 0
        # If not, then the indices of exons below the minimum threshold are kept and only those common 
        # to the previous sample analysed are kept.
        else:
            exons2RMSamp = np.where(sampCounts <= fpmBins[smallestIndex])
            if (len(exons2RMAllSamps) != 0):
                exons2RMAllSamps =  np.intersect1d(exons2RMAllSamps, exons2RMSamp)
            else:
                exons2RMAllSamps = exons2RMSamp

    return(validityStatus, exons2RMAllSamps)      