import logging
import numpy as np
import scipy.stats as st
# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# smoothingCoverageProfile:
# smooth the coverage profile with kernel-density estimate using Gaussian kernel
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
    # - FPMSignal (int): FPM threshold for extracts the exons with the most coverage
    # signals, i.e. with high density and interpretable values.
    # this cutoff is applicable on coverage data from different capture kits
    FPMSignal = 10
    # -binsNb (int); number of bins to create a sufficiently precise range for the FPM
    # in this case the size of a bin is 0.1
    binsNb = FPMSignal * 10

    # FPM data threshold limitation
    sampFragCountsReduced = sampFragCounts[sampFragCounts <= FPMSignal]
    # FPM range creation
    binEdges = np.linspace(0, FPMSignal, num=binsNb)
    # limitation of decimal points to avoid float approximations
    binEdges = np.around(binEdges, 1)

    # scipy.stats.gaussian_kde creates and uses a Gaussian probability density estimate
    # (KDE) from data.
    # the bandwidth determines the width of the Gaussian used to smooth the data when
    # estimating the probability density.
    # it is calculated automatically by scipy.stats.gaussian_kde using Scott's method.
    # bandwidth = n^(-1/(d+4)) * sigma
    #  - n is the number of elements in the data
    #  - d is the dimension of the data
    #  - sigma is the standard deviation of the data
    #
    # - density (scipy.stats.kde.gaussian_kde object): probability density for sampFragCountsReduced
    density = st.kde.gaussian_kde(sampFragCountsReduced)

    # compute density probabilities for each bins in the predefined FPM range
    densityOnFPMRange = density(binEdges)

    return(binEdges, densityOnFPMRange)


###################################
# findLocalMin:
# identifies the first minimum density and the associated index in
# densityOnFPMRnage (identical in binEdges).
# This minimum density corresponds to the threshold separating exons
# with little or no coverage from covered exons.
#
# Args:
#  - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
#
# Returns a tupple (minIndex, minDensity), each variable is created here:
#  - minIndex (int): index with the first lowest density observed
#  - minDensity (float): first lowest density observed
def findLocalMin(densityOnFPMRange):
    #### Fixed parameter
    # threshold number of observed windows with density above the current
    # minimum density (order of magnitude 0.5 FPM)
    subseqWindowSupMin = 5

    #### To Fill:
    # initialize variables for minimum density and index
    minDensity = densityOnFPMRange[0]
    minIndex = 0

    #### Accumulator:
    # counter for number of windows with densities greater than the current
    # minimum density
    counter = 0

    for i in range(1, len(densityOnFPMRange)):
        # current density is lower than the minimum density found so far
        if densityOnFPMRange[i] < minDensity:
            minDensity = densityOnFPMRange[i]
            minIndex = i
            # reset counter
            counter = 0
        # current density is greater than the minimum density found so far
        elif densityOnFPMRange[i] > minDensity:
            counter += 1

        # the counter has reached the threshold, exit the loop
        if counter >= subseqWindowSupMin:
            break

    return (minIndex, minDensity)
