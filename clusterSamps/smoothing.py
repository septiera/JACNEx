import logging
import numpy as np
import KDEpy


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# Given a vector of floats, apply kernel density estimation to obtain a smoothed
# representation of the data.
#
# Args:
# - data: a 1D np.ndarray of floats (>= 0)
# - maxData (float): data values beyond maxData are not plotted (but they ARE
#     used to calculate the bandwidth when using a heuristic)
# - numValues (int): approximate size of the returned x and y, default should be fine
# - bandwidth: one of ('scott', 'silverman', 'ISJ') to use as bandwidth determination
#     heuristic, or a float (to use a fixed bandwidth)
#
# Returns a tuple (x, y, bwValue):
# - x and y are np.ndarrays of the same size representing the smoothed density of data
# - bwValue is the bandwidth value (float) used for the KDE
def smoothData(data, maxData=10, numValues=1000, bandwidth='scott'):
    # if using a bandwidth heuristic ('scott', 'silverman' or 'ISJ'):
    # calculate bwValue using full dataset
    if isinstance(bandwidth, str):
        kde = KDEpy.FFTKDE(bw=bandwidth)
        kde.fit(data).evaluate()
        bwValue = kde.bw
    else:
        bwValue = bandwidth

    # allow 3 bwValues (==stddev since we use gaussian kernels) beyond maxData,
    # to avoid artifactual "dip" of the curve close to maxData
    dataReduced = data[data <= maxData + 3 * bwValue]

    # we have a hard lower limit at 0, this causes some of the weight of close-to-zero
    # values to be ignored (because that weight goes to negative values)
    # this issue can be alleviated by mirroring the data, see:
    # https://kdepy.readthedocs.io/en/latest/examples.html#boundary-correction-using-mirroring
    dataReducedMirrored = np.concatenate((dataReduced, -dataReduced))
    # Compute KDE using bwValue, and twice as many grid points
    (dataRange, density) = KDEpy.FFTKDE(bw=bwValue).fit(dataReducedMirrored).evaluate(numValues * 2)
    # double the y-values to get integral of ~1 on the positive dataRanges
    density = density * 2
    # delete values outside of [0, maxData]
    density = density[np.logical_and(0 <= dataRange, dataRange <= maxData)]
    dataRange = dataRange[np.logical_and(0 <= dataRange, dataRange <= maxData)]
    return(dataRange, density, bwValue)


###################################
# find:
# - the first local min of data, defined as the first data value such that
#   the next windowSize-1 values of data are > data[minIndex] (ignoring
#   stretches of equal values)
# - the first local max of data after minIndex, with analogous definition
#
# Args:
# - data: a 1D np.ndarray of floats
# - windowSize: int, default should be fine if running on y returned by smoothData()
#     with its default numValues
#
# Returns a tuple: (minIndex, maxIndex)
#
# Raise exception if no local min / max is found (eg data is always decreasing, or
# always increasing after minIndex)
def findFirstLocalMinMax(data, windowSize=20):
    # sanity checks
    if windowSize <= 0:
        logger.error("in findFirstLocalMinMax, windowSize must be a positive int, not %s", str(windowSize))
        raise Exception('findLocalMinMax bad args')
    if windowSize > len(data):
        logger.error("findFirstLocalMinMax called with windowSize > len(data), useless")
        raise Exception('findLocalMinMax data too small')

    # find first local min
    minIndex = 0
    minValue = data[minIndex]
    # number of consecutive values >= minValue seen
    thisWindowSize = 1

    for i in range(1, len(data)):
        if data[i] < minValue:
            minIndex = i
            minValue = data[i]
            thisWindowSize = 1
        elif data[i] > minValue:
            thisWindowSize += 1
            if thisWindowSize >= windowSize:
                break
        # else data[i] == minValue, look further

    if thisWindowSize < windowSize:
        raise Exception('findLocalMinMax cannot find a local min')

    # find first local max following minIndex
    maxIndex = minIndex
    maxValue = minValue
    thisWindowSize = 1
    for i in range(minIndex + 1, len(data)):
        if data[i] > maxValue:
            maxIndex = i
            maxValue = data[i]
            thisWindowSize = 1
        elif data[i] < maxValue:
            thisWindowSize += 1
            if thisWindowSize >= windowSize:
                break
        # else continue

    if thisWindowSize < windowSize:
        raise Exception('findLocalMinMax cannot find a max after the min')

    return(minIndex, maxIndex)
