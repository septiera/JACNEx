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
    # order=F should improve performance
    return np.full((numExons, (numSamps * numCN)), -1, dtype=np.float64, order='F')


######################################
# counts2Likelihoods:
# Given a cluster identifier, distribution settings, sample information, and genetic counts,
# this function calculates the likelihoods (probability density values) for various copy number
# scenarios associated with different statistical distributions.
# For each Fragment Per Million (FPM) of an exon within a sample:
#   - CN0: Calculate the Probability Density Function (PDF) using parameters from an
#   exponential distribution fitted to intergenic data (loc = 0, scale = 1 / lambda).
#   These parameters remain constant for all samples in the cohort.
#   - CN2: Compute the PDF based on parameters from a robustly fitted Gaussian
#   distribution, capturing the dominant coverage signal(loc = mean, scale = stdev).
#   These parameters are consistent across all samples in a cluster.
#   - CN1: use the parameters from CN2 while shifting the mean (loc) by
#   a factor of 0.5.
#   This establishes the PDF for CN1, with the same parameters for all samples in a cluster.
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
# - samples (list[strs]): sample identifiers
# - counts (np.ndarray[floats]): normalized fragment counts (FPMs)
# - clust2samps (dict): mapping clusterID to a list of sampleIDs
# - exp_loc[float], exp_scale[float]: parameters of the exponential distribution
# - exonCN2Params (np.ndarray): contains CN2 parameters (gaussian distribution loc=mean, scale=stdev)
#                               and exon filtering status for each cluster.
#                               Dim: nbOfExons * (nbOfClusters * ["loc", "scale", "filterStatus"]).
# - numCNs [int]: 4 copy number status: ["CN0", "CN1", "CN2", "CN3"]
# - numParamsCols [int]: ["loc", "scale", "filterStatus"]
#
# Returns a tupple (clusterID, relevantCols, relevantRows, likelihoodArray):
# - clusterID [str]
# - relevantCols (list[ints]): column indexes in the larger array of all analyzed samples,
#                              where each column corresponds to a specific copy number type (CN0, CN1, CN2, CN3)
#                              for the given clusterID and samples.
#                              These column indexes are used to store the likelihoods in the appropriate
#                              positions of the larger array outside the function.
# - relevantRows (list[ints]): exon indexes in the likelihoodArray corresponding to exons
#                              with filtering status 4 ("Calls") for the given clusterID.
#                              These rows represent the exons for which likelihoods are calculated.
# - likelihoodsArray (np.ndarray[floats]): precomputed likelihoods for each sample and copy number type.
#                                         dim = nbOfRelevantRows * nbOfRelevantCols
def counts2likelihoods(clusterID, samples, counts, clust2samps, exp_loc, exp_scale,
                       exonCN2Params, numCNs, numParamsCols):
    # Fixed parameter:
    # Empirical definition of the alpha parameter based on available data.
    # Achieves a gradual ascending phase of the distribution, ensuring consideration of
    # duplications approximately around gauss_loc*1.5.
    gamma_alpha = 8

    # Convert the dictionary keys to a list and find the index of the clusterID
    clusterIDs = list(clust2samps.keys())
    clusterIndex = clusterIDs.index(clusterID)

    # Get the samples belonging to the specified cluster
    sampsIDs = clust2samps[clusterID]
    sampsIndexes = np.where(np.isin(samples, sampsIDs))[0]

    numSamps = len(sampsIndexes)

    # Extract the relevant columns from exonCN2Params for the specified cluster
    CN2paramsClust = exonCN2Params[:, clusterIndex * numParamsCols:(clusterIndex + 1) * numParamsCols]

    # Create an array of column indexes for each sample,
    relevantCols = np.zeros(len(sampsIndexes) * numCNs, dtype=np.int)

    for sampsIndex in range(len(sampsIndexes)):
        for ci in range(numCNs):
            relevantCols[sampsIndex * numCNs + ci] = sampsIndexes[sampsIndex] * numCNs + ci

    # Find the row indexes where the third column of CN2paramsCluster is equal to 4 ("Calls")
    relevantRows = np.where(CN2paramsClust[:, 2] == 4)[0]

    likelihoodsArray = allocateLikelihoodsArray(numSamps, len(relevantRows), numCNs)

    for rowIndex, exonIndex in enumerate(relevantRows):
        FPMs = counts[exonIndex, sampsIndexes]
        # Get the distribution parameters for this exon
        CN_params = getDistributionParams(exp_loc, exp_scale, CN2paramsClust[exonIndex, 0], CN2paramsClust[exonIndex, 1])

        for ci in range(numCNs):
            distribution, loc, scale = CN_params[ci]

            # Compute the likelihoods for the current CNStatus
            if distribution != scipy.stats.gamma:
                CNLikelihoods = distribution.pdf(FPMs, loc=loc, scale=scale)
            else:
                CNLikelihoods = distribution.pdf(FPMs, a=gamma_alpha, loc=loc, scale=scale)

            # Compute the column indexes
            colIndexes = np.arange(ci, numSamps * numCNs, numCNs)

            # Assign values to likelihoodArray using correct indexes
            likelihoodsArray[rowIndex, colIndexes] = CNLikelihoods

    return (clusterID, relevantCols, relevantRows, likelihoodsArray)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# getDistributionParams
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
# - CN_params (list of tuples): contains the distribution function (Scipy library),
#                               location parameter[float], and scale parameter[float]
#                               for a specific copy number type.
def getDistributionParams(exp_loc, exp_scale, gauss_loc, gauss_scale):
    CN_params = [
        (scipy.stats.expon, exp_loc, exp_scale),  # CN0
        (scipy.stats.norm, gauss_loc * 0.5, gauss_scale),  # CN1
        (scipy.stats.norm, gauss_loc, gauss_scale),  # CN2
        # CN3+ dependent on a gamma distribution:
        #  - 'loc' = gauss_loc + gauss_scale to account for the standard deviation.
        # Prevents overlap of the gamma distribution when the primary Gaussian has
        # a substantial standard deviation, avoiding blending between CN2 and CN3+.
        # - 'scale' = np.log(gauss_loc) to achieve a suitably small theta value
        # for compressing the distribution (normal distribution like), particularly
        # when there is significant spread.
        (scipy.stats.gamma, gauss_loc + gauss_scale, np.log10(gauss_loc)),
    ]
    return CN_params
