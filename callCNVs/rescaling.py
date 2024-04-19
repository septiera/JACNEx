import logging
import numba
import numpy

####### JACNEx modules
import callCNVs.exonProfiling

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#########################
# rescaleClusterFPMAndParams
# Rescale cluster FPM values and their corresponding parameters.
#
# Args:
# - autosomeFPMs (np.ndarray[floats]): FPM values for autosomes.
# - gonosomeFPMs (np.ndarray[floats]): FPM values for gonosomes.
# - CN2Params_A (Dict[str, np.ndarray]): CN2 parameters for autosomes clusters.
# - CN2Params_G (Dict[str, np.ndarray]): CN2 parameters for gonosomes clusters.
# - CN0stdev (float): Standard deviation of CN0.
# - samples (list[str]): sample IDs.
# - clust2samps (Dict[str, List[str]]): Mapping of cluster IDs to sample IDs.
# - fitWith (Dict[str, list[str]]): fitWith (dict): Mapping of cluster IDs to related cluster IDs.
# - clustIsValid (Dict[str, bool]): Validation status of clusters.
#
# Returns a tuple:
# [Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]: Rescaled parameters and FPM values.
def rescaleClusterFPMAndParams(autosomeFPMs, gonosomeFPMs, CN2Params_A, CN2Params_G, CNOstdev,
                               samples, clust2samps, fitWith, clustIsValid):
    FPMsrescal_A = autosomeFPMs.copy()
    FPMsrescal_G = gonosomeFPMs.copy()
    Params_A = {}
    Params_G = {}

    sampIndexMap = callCNVs.exonProfiling.createSampleIndexMap(samples)

    for clusterID in clust2samps:
        # skip processing for invalid clusters
        if not clustIsValid[clusterID]:
            continue

        # get sample indexes for the current cluster and associated fitWith clusters
        try:
            sampsPresence = callCNVs.exonProfiling.getSampPresenceBool(clusterID, clust2samps, sampIndexMap,
                                                                       fitWith, samples, addRelatedClusts=False)
        except Exception as e:
            logger.error("Error in getSampPresenceBool for cluster %s: %s", clusterID, e)
            raise

        # determine the type of chromosome and submit the cluster for processing
        if clusterID.startswith("A"):
            (FPMsrescal_A[:, sampsPresence], Params_A[clusterID]) = processCluster(FPMsrescal_A[:, sampsPresence],
                                                                                   CN2Params_A[clusterID], CNOstdev)
        elif clusterID.startswith("G"):
            (FPMsrescal_G[:, sampsPresence], Params_G[clusterID]) = processCluster(FPMsrescal_G[:, sampsPresence],
                                                                                   CN2Params_G[clusterID], CNOstdev)
        else:
            logger.error("Unknown chromosome type for cluster %s.", clusterID)
            raise

    return (Params_A, Params_G, FPMsrescal_A, FPMsrescal_G)


###############################################################################
############################ PRIVATE FUNCTIONS #################################
###############################################################################
###########################
# processCluster
# Process cluster by rescaling FPM values and corresponding parameters.
#
# Args:
# - clusterFPM (np.ndarray[floats]): FPM values for the cluster.
# - CN2Params (np.ndarray[floats]): CN2 parameters for the cluster [mean, stdev].
# - CN0stdev (float): Standard deviation of CN0.
#
# Returns:
# Tuple[np.ndarray, np.ndarray]: Rescaled FPM values and parameters.
def processCluster(clusterFPM, CN2Params, CN0stdev):
    # Initialize an array to store CN2 parameters with a default value of -1
    paramsRescal = numpy.full_like(CN2Params, -1)
    for ei in range(len(CN2Params)):
        CN2mean = CN2Params[ei][0]
        CN2stdev = CN2Params[ei][1]

        if CN2mean == -1 and CN2stdev == -1:
            continue

        clusterFPM[ei] = rescaleFPM(clusterFPM[ei], CN2mean)

        paramsRescal[ei][0] = rescaleCN2stdev(CN2mean, CN2stdev)

        paramsRescal[ei][1] = rescaleCN0stdev(CN2mean, CN0stdev)

    return (clusterFPM, paramsRescal)


##############################
# rescaleFPM
# Args:
# - exFPM (np.ndarray): Array of FPM values to be rescaled.
# - mu (float): Mean value used for rescaling.
#
# Returns:
# float: Rescaled standard deviation.
@numba.njit
def rescaleFPM(exFPM, mu):
    return exFPM / mu


##############################
# rescaleCN2stdev
# Args:
# - mu (float): Mean value.
# - sigmaCN2 (float): Standard deviation of CN2.
#
# Returns:
# float: Rescaled standard deviation.
@numba.njit
def rescaleCN2stdev(mu, sigmaCN2):
    return sigmaCN2 / mu


##############################
# rescaleCN0stdev
# Args:
# - mu (float): Mean value.
# - sigmaCN2 (float): Standard deviation of CN0.
#
# Returns:
# float: Rescaled standard deviation.
@numba.njit
def rescaleCN0stdev(mu, sigmaCN0):
    return sigmaCN0 / mu
