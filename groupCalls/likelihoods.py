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
# observationCounts2Likelihoods:
# Given a cluster identifier, represented by a list of associated samples (each sample
# represented by a list of fragment counts), this function calculates the likelihoods
# (probability density values) based on the parameters of the continuous distributions
# associated with each copy number type:
# - CN0: Parameters from the exponential distribution.
# - CN2: Parameters from the Gaussian distribution contained in exonCN2Params.
# - CN1: Calculated as meanCN2/2 with the same standard deviation as CN2.
# - CN3+: Calculated as 3*(meanCN2/2) with the same standard deviation as CN2.
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
# - relevantCols (list[ints]): column indices in the larger array of all analyzed samples,
#                              where each column corresponds to a specific copy number type (CN0, CN1, CN2, CN3)
#                              for the given clusterID and samples.
#                              These column indices are used to store the likelihoods in the appropriate
#                              positions of the larger array outside the function.
# - relevantRows (list[ints]): exon indexes in the likelihoodArray corresponding to exons
#                              with filtering status 4 ("Calls") for the given clusterID.
#                              These rows represent the exons for which likelihoods are calculated.
# - likelihoodArray (np.ndarray[floats]): precomputed likelihoods for each sample and copy number type.
#                                         dim = nbOfRelevantRows * nbOfRelevantCols
def observationCounts2Likelihoods(clusterID, samples, counts, clust2samps, exp_loc, exp_scale,
                                  exonCN2Params, numCNs, numParamsCols):

    # Convert the dictionary keys to a list and find the index of the clusterID
    clusterIDs = list(clust2samps.keys())
    clusterIndex = clusterIDs.index(clusterID)

    # Get the samples belonging to the specified cluster
    sampsIDs = clust2samps[clusterID]
    sampsIndexes = np.where(np.isin(sampsIDs, samples))[0]

    numSamps = len(sampsIndexes)

    # Extract the relevant columns from exonCN2Params for the specified cluster
    CN2paramsClust = exonCN2Params[:, clusterIndex * numParamsCols:(clusterIndex + 1) * numParamsCols]

    # Create an array of column indices for each sample,
    # np.tile: repeating the pattern [0, 1, 2, 3] numSamps times
    relevantCols = np.tile(np.arange(numCNs), numSamps) + np.repeat(sampsIndexes, numCNs) * numCNs
    # relevantCols = [ci + sampsIndex * numCNs for sampsIndex in sampsIndexes for ci in range(numCNs)]

    # Find the row indices where the third column of CN2paramsCluster is equal to 4 ("Calls")
    relevantRows = np.where(CN2paramsClust[:, 2] == 4)[0]

    likelihoodArray = allocateLikelihoodsArray(numSamps, len(relevantRows), numCNs)

    for rowIndex, exonIndex in enumerate(relevantRows):
        FPMs = counts[exonIndex, sampsIndexes]
        # Get the distribution parameters for this exon
        CN_params = getDistributionParams(exp_loc, exp_scale, CN2paramsClust[exonIndex, 0], CN2paramsClust[exonIndex, 1])

        for ci in range(numCNs):
            distribution, loc, scale = CN_params[ci]

            # Compute the likelihoods for the current CNStatus using vectorized operation
            CNLikelihoods = distribution.pdf(FPMs, loc=loc, scale=scale)

            # Compute the column indices using vectorized operations
            colIndices = np.arange(ci, numSamps * numCNs, numCNs)

            # Assign values to likelihoodArray using correct indexes
            likelihoodArray[rowIndex, colIndices] = CNLikelihoods

    return (clusterID, relevantCols, relevantRows, likelihoodArray)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# getDistributionParams
# Get the parameters of the continuous distributions for each copy number type.
#
# Args:
# - exp_loc [float]: Location parameter for the exponential distribution (CN0).
# - exp_scale [float]: 1 / lambda parameter for the exponential distribution (CN0).
# - gauss_loc [float]: Mean parameter for the Gaussian distribution (CN2).
# - gauss_scale [float]: Standard deviation parameter for the Gaussian distribution (CN2).
# Returns:
# - CN_params (tupple): contains the distribution function, location parameter,
#                       and scale parameter for a specific copy number type.
def getDistributionParams(exp_loc, exp_scale, gauss_loc, gauss_scale):
    CN_params = [
        (scipy.stats.expon, exp_loc, exp_scale),  # CN0
        (scipy.stats.norm, gauss_loc / 2, gauss_scale),  # CN1
        (scipy.stats.norm, gauss_loc, gauss_scale),  # CN2
        (scipy.stats.norm, 3 * (gauss_loc / 2), gauss_scale),  # CN3
    ]
    return CN_params
