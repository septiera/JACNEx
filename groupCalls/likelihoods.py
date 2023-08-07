import logging
import numpy as np

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
#
# Returns a tupple (sampLikelihoodsColInd, sampLikelihoodArray):
# - sampLikelihoodsColInd (list[ints]): Column indexes to fill in the large likelihood array.
# - sampLikelihoodArray (np.ndarray[floats]): Likelihoods for all CN types for cluster samples.


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################


