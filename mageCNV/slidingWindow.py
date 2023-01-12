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
#
# Args:
#  - densities (np.ndarray[float]): exons densities obtained from a FPM range
#  - windowSize (int): number of bins in a window
#
# Returns:
#  - densityMeans (np.ndarray[float]): mean density for each window covered. dim=len(densities)
def smoothingCoverageProfile(densities, windowSize):
    # check if window_size is even, if so add 1 to make it odd
    if windowSize % 2 == 0:
        windowSize += 1

    # To fill:
    densityMeans = np.zeros(len(densities), dtype=np.float)

    # an index (i) is the middle of the windowSize (e.g windowSize=5, index=3)
    for i in range(0, len(densities)):
        # case 1: the starting indexes don't include all the window bins
        if (i <= windowSize // 2):
            # Initialization of the sum at index 0.
            # Only the window bins larger than the index are considered
            # e.g windowSize=5, middle window=0, sum -> densities[0,1,2]
            if (i == 0):
                # recall correct values range selection in an np.ndarray[0:n+1]
                rollingSum = sum(densities[:(windowSize // 2) + 1])
                # add index to denominator (+1) for mean computation
                densityMeans[i] = rollingSum / ((windowSize // 2) + 1)
            # the following indexes keep the previous density values
            # e.g windowSize=5, middle window=1, sum -> densities[0,1,2,3]
            else:
                # add new density value in accordance with the window offset
                rollingSum += densities[i + (windowSize // 2)]
                # the index value is equal to the number of values kept so it's
                # added to the denominator for mean computation
                densityMeans[i] = rollingSum / (i + windowSize // 2 + 1)

        # case 2: all window bins can be taken
        # len(densities) is a count so doesn't consider the index 0 (-1)
        elif (i <= ((len(densities) - 1) - (windowSize // 2))):
            rollingSum -= densities[i - (windowSize // 2) - 1]
            rollingSum += densities[i + (windowSize // 2)]
            densityMeans[i] = rollingSum / windowSize

        # case 3: the latest indexes don't include all the window bins
        else:
            # remove old density value in accordance with the window offset
            rollingSum -= densities[i - (windowSize // 2) - 1]
            # the denominator is the number of bins remaining
            # "(len(density)-i" is a contraction of "(len(density)-1)-(i+1)"
            densityMeans[i] = rollingSum / (windowSize // 2 + (len(densities) - i))

    return(densityMeans)


###################################
# findLocalMin:
# find the threshold of the minimum density averages before an increase
# allowing to differentiate covered and uncovered exons.
#
# Args:
#  - densityMeans (list[float]): average density for each window covered
#
# Returns a tupple (minIndex, minDensity), each variable is created here:
#  - minIndex (int): index from densityMean associated with the first lowest
#    observed average
#  - minDensitySum (float): first lowest observed average
def findLocalMin(densityMeans):
    #### Fixed parameter
    # threshold number of observed windows with density average above the current
    # minimum average (order of magnitude 0.5 FPM)
    subseqWindowSupMin = 5

    #### To Fill:
    # initialize variables for minimum density mean and index
    minMean = densityMeans[0]
    minIndex = 0

    #### Accumulator:
    # counter for number of windows with densities greater than the current
    # minimum density
    counter = 0

    for i in range(1, len(densityMeans)):
        # current density is lower than the minimum density found so far
        if densityMeans[i] < minMean:
            minMean = densityMeans[i]
            minIndex = i
            # reset counter
            counter = 0
        # current density is greater than the minimum density found so far
        elif densityMeans[i] > minMean:
            counter += 1

        # the counter has reached the threshold, exit the loop
        if counter >= subseqWindowSupMin:
            break

    return (minIndex, minMean)
