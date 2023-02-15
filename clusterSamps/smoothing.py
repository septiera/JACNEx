import logging
import numpy as np
import scipy.stats


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# smoothingCoverageProfile:
# Smooth the coverage profile with kernel-density estimate using Gaussian kernel
# Limitation of the count data used to a FPM threshold (10) for extracts the exons
# with the most coverage signals, i.e. with high density and interpretable values.
# This cutoff is applicable on coverage data from different capture kits.
# Definition of a sufficiently precise FPM range (0.1) to deduce the densities of the exons.
# scipy.stats.gaussian_kde creates and uses a Gaussian probability density estimate
# (KDE) from data.
# The bandwidth determines the width of the Gaussian used to smooth the data when
# estimating the probability density.
# It's calculated automatically by scipy.stats.gaussian_kde using Scott's method.
# bandwidth = n^(-1/(d+4)) * sigma
#  - n is the number of elements in the data
#  - d is the dimension of the data
#  - sigma is the standard deviation of the data
#
# Args:
# - sampFragCounts (np.ndarray[float]): FPM by exon for a sample, dim=NbExons
#
# Returns:
# - binEdges (np.ndarray[floats]): FPM range
# - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
#   dim= len(binEdges)
def smoothingCoverageProfile(sampFragCounts):

    #### Fixed parameters:
    # - FPMSignal (int): FPM threshold
    FPMSignal = 10
    # -binsNb (int): number of bins to create a sufficiently precise range for the FPM
    # in this case the size of a bin is 0.1
    binsNb = FPMSignal * 10

    # FPM data threshold limitation
    sampFragCountsReduced = sampFragCounts[sampFragCounts <= FPMSignal]
    # FPM range creation
    binEdges = np.linspace(0, FPMSignal, num=binsNb)
    # limitation of decimal points to avoid float approximations
    binEdges = np.around(binEdges, 1)

    # - density (scipy.stats.kde.gaussian_kde object): probability density for sampFragCountsReduced
    # Beware all points are evaluated
    density = scipy.stats.kde.gaussian_kde(sampFragCountsReduced)

    # compute density probabilities for each bins in the predefined FPM range
    densityOnFPMRange = density(binEdges)

    return(binEdges, densityOnFPMRange)


###################################
# find:
# - the first local min of data, defined as the first data value such that
#   the next windowSize-1 values of data are > data[minIndex] (ignoring
#   stretches of equal values)
# - the first local max of data after minIndex, with analogous definition
#
# Args:
# - data: a 1D np.ndarray of floats
# - windowSize: int, with default value
#
# Returns a tuple: (minIndex, maxIndex)
#
# Raise exception if no local min / max is found (eg data is always decreasing, or
# always increasing after minIndex)
def findFirstLocalMinMax(data, windowSize=6):
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
        logger.warning("findFirstLocalMinMax can't find local min, doesn't data ever increase?")
        raise Exception('findLocalMinMax cannot find a min')

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
        logger.warning("findFirstLocalMinMax can't find local max, doesn't data ever decrease after minIndex?")
        raise Exception('findLocalMinMax cannot find a max after the min')

    return(minIndex, maxIndex)
