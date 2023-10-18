import logging
import numpy as np
import scipy.stats

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
######################################
# allocateLikelihoodsArray:
#
# Args:
# - numSamps [int]: number of samples to process
# - numExons [int]: number of exons to process
# - numCN [int]: number of copy numbers to process (default = 4 , CN0,CN1,CN2,CN3+)
#
# Returns an float array with -1, allocated likelihoods for each
# individual and copy number combination for each exons.
# dim = nbOfExons * (nbOfSamps * nbOfCN)
def allocateLikelihoodsArray(numSamps, numExons, numCN):
    # order=C should improve performance
    return np.full((numExons, (numSamps * numCN)), -1, dtype=np.float64, order='C')


######################################
# counts2Likelihoods:
# Given a cluster identifier, distribution settings, sample information, and genetic counts,
# this function calculates the likelihoods (probability density values) for various copy number
# scenarios associated with different statistical distributions.
# For each Fragment Per Million (FPM) of an exon within a sample:
#   - CN0: Calculate the Probability Density Function (PDF) using parameters from an
#   exponential distribution fitted to intergenic data (loc = 0, scale = 1 / lambda).
#   - CN2: Compute the PDF based on parameters from a robustly fitted Gaussian
#   distribution, capturing the dominant coverage signal(loc = mean, scale = stdev).
#   - CN1: use the parameters from CN2 while shifting the mean (loc) by
#   a factor of 0.5..
#   - CN3+: Addressing scenarios where copy numbers exceed 2, parameters from
#   the CN2 Gaussian distribution are leveraged to empirically establish the parameters for
#   a distribution with a heavy tail.
#   The chosen heavy-tailed distribution is a gamma distribution. Its parameters are:
#       - alpha [integer], representing shape (curvature) in SciPy.
#       If alpha > 1, heavy tail; if alpha = 1, exponential-like; alpha < 1, light tail.
#       - theta [float] stretches or compresses distribution (scale in Scipy).
#       Higher theta expands, lower compresses.
#       Also, 'loc' parameter in SciPy shifts distribution along x-axis without changing
#       shape or stretch.
# This function is an advanced step of the Hidden Markov Model (HMM) as it
# precomputes the emission probabilities from the observations(FPMs).
#
# Args:
# - clusterID [str]
# - samp2Index (dict): keys == sampleIDs, values == exonsFPM samples column indexes
# - exonsFPM (np.ndarray[floats]): normalized fragment counts (FPMs)
# - clust2samps (dict): mapping clusterID to a list of sampleIDs
# - exp_loc[float], exp_scale[float]: parameters of the exponential distribution
# - exMetrics (dict): keys == clusterIDs, values == np.ndarray [floats]
#                     Dim = nbOfExons * ["loc", "scale", "filterStatus"].
# - numCNs [int]: 4 copy number status: ["CN0", "CN1", "CN2", "CN3"]
# - chromType [str]
#
# Returns a tupple (clusterID, likelihoodArray, chromType):
# - clusterID [str]
# - chromType [str]: type of chromosome where exons are analysed :"A" for autosomes,
#                    "G" for gonosomes.
# - likelihoodClustDict : keys == sampleID, values == np.ndarray(nbExons * nbCNStates)

def counts2likelihoods(clusterID, samp2Index, exonsFPM, clust2samps, exp_loc, exp_scale,
                       exMetrics, numCNs, chromType):

    try:
        logger.debug("process cluster %s", clusterID)

        # IDs and indexes (in "samples" and "counts" columns) of samples from current cluster
        sampsIDs = list(clust2samps[clusterID])
        sampsIndexes = [samp2Index[samp] for samp in sampsIDs]

        # np.array 2D dim = (NbExons * [loc[float], scale[float], filterStatus[int]])
        clusterMetrics = exMetrics[clusterID]

        likelihoodClustDict = {}
        for samp in sampsIDs:
            likelihoodClustDict[samp] = np.full((exonsFPM.shape[0], numCNs), -1, dtype=np.float128, order='C')

        for exonIndex in range(len(clusterMetrics)):
            if clusterMetrics[exonIndex, 2] != 4:
                continue
            gauss_loc = clusterMetrics[exonIndex, 0]
            gauss_scale = clusterMetrics[exonIndex, 1]

            distribution_functions = getDistributionObjects(exp_loc, exp_scale, gauss_loc, gauss_scale)

            for ci in range(numCNs):
                pdf_function, loc, scale, shape = distribution_functions[ci]
                exFPM = exonsFPM[exonIndex, sampsIndexes]
                # np.ndarray 1D float: set of pdfs for all samples
                # scipy execution speed up
                if shape is not None:
                    res = pdf_function(exFPM, loc=loc, scale=scale, a=shape)
                else:
                    res = pdf_function(exFPM, loc=loc, scale=scale)

                for si in range(len(sampsIndexes)):
                    likelihoodClustDict[sampsIDs[si]][exonIndex, ci] = res[si]

        return (chromType, clusterID, likelihoodClustDict)

    except Exception as e:
        logger.error("Likelihoods failed for cluster %s - %s", clusterID, repr(e))
        raise Exception(clusterID)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# getDistributionObjects
# Defines parameters for four types of distributions (CN0, CN1, CN2, CN3+),
# involving exponential, normal, and gamma distributions.
# For CN3+, the parameters are empriricaly adjusted to ensure compatibility between
# Gaussian and gamma distributions.
#
# Args:
# - exp_loc [float]: Location parameter for the exponential distribution (CN0).
# - exp_scale [float]: 1 / lambda parameter for the exponential distribution (CN0).
# - gauss_loc [float]: Mean parameter for the Gaussian distribution (CN2).
# - gauss_scale [float]: Standard deviation parameter for the Gaussian distribution (CN2).
# Returns:
# - CN_params(list of list): contains distribution objects from Scipy representing
#                     different copy number types (CN0, CN1, CN2, CN3+).
#                     Parameters vary based on distribution type (ordering: loc, scale, shape).
def getDistributionObjects(exp_loc, exp_scale, gauss_loc, gauss_scale):

    # shifting Gaussian mean for CN1
    gaussShiftLoc = gauss_loc * 0.5

    # CN3+ dependent on a gamma distribution:
    #  - 'a': Empirical definition of the alpha parameter based on available data.
    # Achieves a gradual ascending phase of the distribution, ensuring consideration of
    # duplications approximately around gauss_loc*1.5.
    #  - 'loc' = gauss_loc_plus_scale take into account the standard deviation.
    # Prevents overlap of the gamma distribution when the primary Gaussian has
    # a substantial standard deviation, avoiding blending between CN2 and CN3+.
    #  - 'scale' = log_gauss_loc_plus_1 adapts to the data by scaling the distribution.
    # "+1" prevents division by zero issues for means <= 1, and using log encloses
    # the scale around 1, creating a distribution similar to a Gaussian.
    gamma_shape = 8
    # Calculate the sum of gauss_loc and gauss_scale once
    gauss_locAddScale = gauss_loc + gauss_scale
    # Calculate the logarithm of gauss_loc_plus_scale + 1 once
    gauss_logLocAdd1 = np.log10(gauss_locAddScale + 1)

    CN_params = [(scipy.stats.expon.pdf, exp_loc, exp_scale, None),  # CN0
                 (scipy.stats.norm.pdf, gaussShiftLoc, gauss_scale, None),  # CN1
                 (scipy.stats.norm.pdf, gauss_loc, gauss_scale, None),  # CN2
                 (scipy.stats.gamma.pdf, gauss_locAddScale, gauss_logLocAdd1, gamma_shape)]  # CN3+
    return CN_params
