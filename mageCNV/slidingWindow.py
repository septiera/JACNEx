import logging
import numpy as np
# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# smoothingCoverageProfile:
# smooth the coverage profile from a sliding window.
# Takes an odd window size to obtain indices consistent with the calculated density averages.
# Args:
#  - densities (np.array[float]): number of elements in the bin / (bin width * total number of elements)
#  - windowSize (int): number of bins in a window
# Returns a tupple (middleIndexes , densityMeans), each variable is created here:
#  - middleWindowIndexes (list[int]): the middle indices of each window
#    (in agreement with the fpmBins indexes)
#  - densityMeans (list[float]): mean density for each window covered
def smoothingCoverageProfile(densities, windowSize):
    # check if window_size is even, if so add 1 to make it odd
    if windowSize % 2 == 0:
        windowSize += 1

    # Accumulators:
    middleWindowIndexes = []
    densityMeans = []

    # calculate the density sum of the first window
    initial_sum = sum(densities[:windowSize]) 
    # storage of the density mean and indice associated with the first window
    densityMeans.append(initial_sum / windowSize)
    middleWindowIndexes.append((windowSize // 2) + 1)  # ceil not floor

    # calculate density mean for rest of the windows
    for i in range(1, len(densities) - windowSize + 1):
        # add middle index of each window to the list
        middleWindowIndexes.append(i + (windowSize - 1 // 2) + 1)
        # remove the first element of the previous window
        initial_sum -= densities[i - 1]
        # add the last element of the next window
        initial_sum += densities[i + windowSize - 1]
        # calculate the density mean of the current window
        densityMeans.append(initial_sum / windowSize)

    return(middleWindowIndexes, densityMeans)


###################################
# findLocalMin:
# recover the threshold of the minimum density averages before an increase
# allowing to differentiate covered and uncovered exons.
# Args:
# all arguments are from the smoothingCoverProfile function.
#  - middleWindowIndexes (list[int]): the middle indices of each window
#    (in agreement with the fpmBins indexes)
#  - densityMeans (list[float]): average density for each window covered
# Returns a tupple (minIndex, minDensity), each variable is created here:
#  - minIndex (int): index associated with the first lowest observed average
#   (in agreement with the fpmBins indexes)
#  - minDensitySum (float): first lowest observed average
def findLocalMin(middleWindowIndexes, densityMeans):
    # threshold for number of windows observed with densities mean greater than the
    # current minimum density mean
    subseqWindowSupMin = 20

    # To Fill:
    # initialize variables for minimum density mean and index
    minDensityMean = densityMeans[0]
    minDensityIndex = 0

    # Accumulator:
    # counter for number of windows with densities greater than the current
    # minimum density
    counter = 0

    for i in range(1, len(densityMeans)):
        # current density is lower than the minimum density found so far
        if densityMeans[i] < minDensityMean:
            minDensityMean = densityMeans[i]
            minDensityIndex = i
            # reset counter
            counter = 0
        # current density is greater than the minimum density found so far
        elif densityMeans[i] > minDensityMean:
            counter += 1

        # the counter has reached the threshold, exit the loop
        if counter >= subseqWindowSupMin:
            break

    minIndex = middleWindowIndexes[minDensityIndex]
    return (minIndex, minDensityMean)
