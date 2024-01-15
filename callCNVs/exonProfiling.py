import logging
import math
import numba
import numpy as np
import scipy.stats
from concurrent.futures import ProcessPoolExecutor

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
# calcCN0Params
# Calculates the parameters "hnorm_loc" and "hnorm_scale" associated with the
# fitted half normal distribution from the intergenic count data.
# It also calculates the "uncaptThreshold".
# This is the FPM (Fragments Per Million) limit below which exons are considered
# "uncaptured" due to insufficient sequencing or capture.
# It's derived from the intergenic fragment distribution and set at the 95th
# percentile of the associated half normal distribution (CN0).
#
# Args:
# - intergenicsFPM (np.ndarray[floats]): count data from intergenic regions.
#
# Returns a tuple (hnormloc, hnormscale, uncaptThreshold):
# - hnormloc [float]: location parameter of the fitted half normal distribution.
# - hnormscale [float]: scale parameter of the fitted half normal distribution.
# - uncaptThreshold [float]: FPM threshold.
def calcCN0Params(intergenicsFPM):
    # half normal CDF fraction beyond which we truncate this distribution
    fracCDFHnorm = 0.95
    try:
        (hnormloc, hnormscale) = fitHalfNormal(intergenicsFPM)
    except Exception as e:
        logger.error("fitHalfNormal failed: %s", repr(e))
        raise

    # Calculate the FPM limit below which exons are considered "uncaptured"
    uncaptThreshold = scipy.stats.halfnorm.ppf(fracCDFHnorm, loc=hnormloc, scale=hnormscale)

    return (hnormloc, hnormscale, uncaptThreshold)


##############################################################
# calcCN2Params
# Calculates CN2 parameters for both autosomes and gonosomes using input FPMs.
# Maps sample names to indices for efficient lookup.
# Determines the number of clusters for parallel processing.
# Iterates over clusters, processing each valid cluster in parallel.
#
# Args:
# - autosomeFPMs (np.ndarray[floats]): Autosome fragment counts.
# - gonosomeFPMs (np.ndarray[floats]): Gonosome fragment counts.
# - samples (list[str]): sample names.
# - uncaptThreshold (float): Threshold for considering exons as uncaptured(<).
# - clust2samps (dict): Mapping of cluster IDs to sample lists.
# - fitWith (dict): Mapping of cluster IDs to related cluster IDs.
# - clustIsValid (dict): Indicates validity of each cluster.
# - jobs (int): Number of jobs for parallel processing.
#
# Returns a tuple (CN2Params_A, CN2Params_G):
# - CN2Params_A (dict): for autosomal clusters, key==clusterID,
#                       value==np.ndarray[floats] dim=NbOfExons*["loc", "scale"].
# - CN2Params_G (dict): same as CN2Params_A but for gonosomal clusters.
def calcCN2Params(autosomeFPMs, gonosomeFPMs, samples, uncaptThreshold,
                  clust2samps, fitWith, clustIsValid, jobs):
    # initialize output dictionaries
    CN2Params_A = {}
    CN2Params_G = {}

    # convert the list of samples to a dictionary mapping sample name to its index
    # for O(1) lookup time
    sampIndexMap = {sample: i for i, sample in enumerate(samples)}

    # determine the number of clusters to process in parallel, based on available jobs
    paraClusters = min(math.ceil(jobs / 2), len(clust2samps))
    logger.info("%i new clusters => will process %i in parallel", len(clust2samps), paraClusters)

    # set up a pool of workers for parallel processing
    with ProcessPoolExecutor(paraClusters) as pool:
        # iterate over all clusters and submit processing tasks
        for clusterID in clust2samps:

            # skip processing for invalid clusters
            if not clustIsValid[clusterID]:
                logger.warning("cluster %s is invalid, low sample number %i, skipping it.",
                               clusterID, len(clust2samps[clusterID]))
                continue

            # get sample indexes for the current cluster and associated fitWith clusters
            try:
                sampsInd = getSampIndexes(clusterID, clust2samps, sampIndexMap, fitWith)
            except Exception as e:
                logger.error("Error in getSampIndexes for cluster %s: %s", clusterID, e)
                raise

            # Determine the type of chromosome and submit the cluster for processing
            if clusterID.startswith("A"):
                processCN2Cluster(clusterID, autosomeFPMs[:, sampsInd], uncaptThreshold, CN2Params_A, pool)
            elif clusterID.startswith("G"):
                processCN2Cluster(clusterID, gonosomeFPMs[:, sampsInd], uncaptThreshold, CN2Params_G, pool)
            else:
                logger.error("Unknown chromosome type for cluster %s.", clusterID)
                raise

    return (CN2Params_A, CN2Params_G)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################################################
# fitHalfNormal
# Given a fragment count array (dim = NBOfExons*NBOfSamps),
# fits a half normal distribution, fixing location to 0.
#
# Args:
# - intergenicsFPM (np.ndarray[floats]): count array (FPM normalized)
#
# Returns a tuple (loc, scale):
# - hnormloc [float], hnormscale[float]: parameters of the half normall distribution
def fitHalfNormal(intergenicsFPM):
    # Fit a half normal distribution,
    # enforces the "loc" parameter to be 0 because our model requires the distribution
    # to start at zero.
    # scale = std_deviation
    # f(x) = (1 / (scale * sqrt(2 * pi))) * exp(-x**2 / (2 * scale**2))
    hnormloc, hnormscale = scipy.stats.halfnorm.fit(intergenicsFPM, floc=0)

    return (hnormloc, hnormscale)


#############################################################
# getSampIndexes
# consolidates the indexes of sample IDs that are associated with a specific cluster
# and any additional clusters specified to 'fit with' the primary cluster.
#
# Args:
# -clusterID [str]: identifier of the primary cluster being processed.
# -clust2samps (dict): key==clusterID, value==list of sampleIDs.
# -samp2Index (dict): key==sampleID, value==sample column index in 'counts' array.
# -fitWith(dict): key==clusterID, value==list of clusterIDs.
#
# Returns a list of samples indexes from 'counts' array.
def getSampIndexes(clusterID, clust2samps, samp2Index, fitWith):
    # Gather all the sample IDs for the current cluster and associated fitWith clusters.
    allSamps = set(clust2samps[clusterID])
    for clustID in fitWith[clusterID]:
        allSamps.update(clust2samps[clustID])

    # Convert the sample IDs to indexes using samp2Index.
    allSampsInd = [samp2Index[sample] for sample in allSamps]

    return allSampsInd


##############################################################
# processCN2Cluster
# Submits a task to process a single cluster's FPM data to calculate CN2 parameters.
# Submits a task to the processing pool.
# On task completion, triggers a callback to update the parameters.
#
# Args:
# - clusterID (str): Identifier of the cluster.
# - FPMs (np.ndarray): Fragment counts for the cluster.
# - uncaptThreshold (float): Threshold for uncaptured exons.
# - CN2Params (dict): Dictionary to store CN2 parameters.
# - pool (ProcessPoolExecutor): Pool of workers for parallel processing.
def processCN2Cluster(clusterID, FPMs, uncaptThreshold, CN2Params, pool):
    future_res = pool.submit(computeClusterCN2Params, FPMs, uncaptThreshold)
    future_res.add_done_callback(lambda future: updateCN2Params(future, CN2Params, clusterID))


##############################################################
# updateCN2Params
# Updates the CN2 parameters dictionary with results from processed clusters.
# Checks for exceptions in the processing task.
# Updates the CN2Params dictionary with new data from the completed task.
# Args:
# - future_clusterMetrics (Future): Future object from completed task.
# - CN2Params (dict): Dictionary to be updated with CN2 parameters.
# - clusterID (str): Identifier of the processed cluster.
#
# Output:
# Updates dictionary with new CN2 parameters for the processed cluster.
def updateCN2Params(future_clusterMetrics, CN2Params, clusterID):
    e = future_clusterMetrics.exception()
    if e is not None:
        logger.warning("Analysis failed for cluster %s: %s.", clusterID, str(e))
    else:
        CN2ParamsArray = future_clusterMetrics.result()
        CN2Params[clusterID] = CN2ParamsArray
        logger.info("Completed analysis for cluster %s.", clusterID)


#############################################################
# computeClusterCN2Params
# Processes exon data for a given cluster, apply filtering criteria, and compute CN2
# parameters for each exon.
#
# Args:
# - exonFPMs (np.ndarray[floats]): normalised counts from exons
# - uncaptThreshold [float]: threshold for determining if an exon is captured or not.
#
# Returns:
# - CN2Params (np.ndarray[floats]): dim: NBofExons x NBofMetrics.
#                                        Metrics include 'loc' (mean) and 'scale' (standard deviation) for each exon.
def computeClusterCN2Params(exonFPMs, uncaptThreshold):
    # Possible filtering states for exons
    filterStates = ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"]

    # Define parameter identifiers for CN2 metrics
    paramsID = ["loc", "scale"]

    # Initialize an array to store CN2 parameters with a default value of -1
    CN2Params = np.full((exonFPMs.shape[0], len(paramsID)), -1, dtype=np.float64)

    # Vector to keep track of the filter state for each exon
    filterStatesVec = np.zeros(exonFPMs.shape[0], dtype=int)

    # Process each exon to determine its filter state and compute CN2 parameters
    for exonIndex, exFPMs in enumerate(exonFPMs):
        try:
            # Apply filtering and update parameters for the current exon
            filterState, exonCN2Params = assessExonAndComputeCN2Metrics(exFPMs, uncaptThreshold, paramsID)
            filterStatesVec[exonIndex] = filterStates.index(filterState)
            if filterState == 'call':
                # Update CN2ParamsArray only for exon calls
                CN2Params[exonIndex] = exonCN2Params
        except Exception as e:
            logger.error("evaluateAndComputeExonMetrics failed at exon index %i: %s", exonIndex, repr(e))
            raise

    return CN2Params


#############################################################
# assessExonAndComputeCN2Metrics
# processes an exon by applying a series of filters to determine its filter status.
# Specs:
# 1) Check if the exon is not captured (median coverage = 0).
# 2) Fit a robust Gaussian distribution (CN2) to the exon FPM values.
# 3) Check if the fitted Gaussian overlaps the threshold associated with the
#    uncaptured exon profile.
# 4) Check if the sample contribution rate to the Gaussian is too low (<50%).
# 5) If any of the above filters is triggered, return the corresponding filter status.
# 6) If the exon passes filters, the clusterParamsArray is updated with the
#    computed parameters of the Gaussian distribution for the exon.
#
# Args:
# -exFPMs (numpy.ndarray[floats]): FPM values for the exon.
# -unCaptFPMLimit [float]: threshold value below which exons are considered uncaptured.
# -paramsID (list[strs]): List of parameter identifiers, e.g., ["loc", "scale"].
#
# Returns a tuple containing the filter status as a string and an array of computed CN2 parameters.
# The CN2 parameters array contains 'loc' (mean) and 'scale' (standard deviation) if calculated.
def assessExonAndComputeCN2Metrics(exFPMs, unCaptFPMLimit, paramsID):
    # CN2 parameters array with default NaN values
    exonCN2Params = np.full((1, len(paramsID)), np.nan)

    # Filter n°1: Check if exon is not captured, indicated by zero median coverage
    if filterUncapturedExons(exFPMs):
        return ("notCaptured", exonCN2Params)

    # Attempt robust Gaussian fitting to calculate CN2
    # Filter n°2: Check if fitting is impossible, indicated by median close to zero
    try:
        (gaussian_loc, gaussian_scale) = fitRobustGaussian(exFPMs)
    except Exception as e:
        if str(e) == "cannot fit":
            return ("cannotFitRG", exonCN2Params)
        else:
            raise Exception(f"fitRobustGaussian failed: {repr(e)}")

    # If none of the two first filters applied, update the CN2 parameters with calculated values
    exonCN2Params[0, paramsID.index("loc")] = gaussian_loc
    exonCN2Params[0, paramsID.index("scale")] = gaussian_scale

    ### Filter n°3: Check if fitted gaussian overlaps the uncaptured exon profile threshold
    if filterZscore(gaussian_loc, gaussian_scale, unCaptFPMLimit):
        return ("RGClose2LowThreshold", exonCN2Params)

    ### Filter n°4: Check if the sample's contribution to the gaussian is too low (less than 50%)
    if filterSampsContrib2Gaussian(gaussian_loc, gaussian_scale, exFPMs):
        return ("fewSampsInRG", exonCN2Params)

    return ("call", exonCN2Params)


#############################################################
# filterUncapturedExons
# Given a FPM counts from an exon, calculating the median coverage and filter
# several possible cases:
#   - all samples in the cluster haven't read capture for the current exon
#   - more than 2/3 of the samples have no capture.
# Warning: Potential presence of homodeletions. We have chosen don't call
# them because they affect too many samples
#
# Args:
# - exFPMs (numpy.ndarray[floats]): FPM counts from an exon
#
# Returns "True" if exon doesn't pass the filter otherwise "False"
@numba.njit
def filterUncapturedExons(exFPMs):
    medianFPM = np.median(exFPMs)
    if medianFPM == 0:
        return True
    else:
        return False


#############################################################
# fitRobustGaussian
# Fits a single principal Gaussian component around a starting guess point
# in a 1-dimensional Gaussian mixture of unknown components with EM algorithm
# script found to :https://github.com/hmiemad/robust_Gaussian_fit (v01_2023)
#
# Args:
# - X (np.array): A sample of 1-dimensional mixture of Gaussian random variables
# - mean (float, optional): Expectation. Defaults to None.
# - stdev (float, optional): Standard deviation. Defaults to None.
# - bandwidth (float, optional): Hyperparameter of truncation. Defaults to 2.
# - eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.
#
# Returns a tuple (mu, stdev), parameters of the normal fitted,
# may return an exception if the fit cannot be achieved.
def fitRobustGaussian(X, mean=None, stdev=None, bandwidth=2.0, eps=1.0e-5):
    if mean is None:
        # median is an approach as robust and naïve as possible to Expectation
        mean = np.median(X)
    mu_0 = mean + 1

    if stdev is None:
        # rule of thumb
        stdev = np.std(X) / 3
    sigma_0 = stdev + 1

    bandwidth_truncated_normal_sigma = truncated_integral_and_sigma(bandwidth)

    while abs(mean - mu_0) + abs(stdev - sigma_0) > eps:
        # loop until tolerence is reached
        """
        create a uniform window on X around mu of width 2*bandwidth*sigma
        find the mean of that window to shift the window to most expected local value
        measure the standard deviation of the window and divide by the standard deviation of a truncated Gaussian distribution
        measure the proportion of points inside the window, divide by the weight of a truncated Gaussian distribution
        """
        Window = np.logical_and(X - mean - bandwidth * stdev < 0, X - mean + bandwidth * stdev > 0)

        # condition to identify exons with points arround at the median
        if Window.any():
            mu_0, mean = mean, np.average(X[Window])
            var = np.average(np.square(X[Window])) - mean**2
            sigma_0, stdev = stdev, np.sqrt(var) / bandwidth_truncated_normal_sigma
        # no points arround the median
        # e.g. exon where more than 1/2 of the samples have an FPM = 0.
        # A Gaussian fit is impossible => raise exception
        else:
            raise Exception("cannot fit")
    return (mean, stdev)


#############################################################
# normal_erf
# Computes Gauss error function (erf)
# used by fitRobustGaussian function
def normal_erf(x, mean=0, sigma=1, depth=50):
    ele = 1.0
    normal = 1.0
    x = (x - mean) / sigma
    erf = x
    for i in range(1, depth):
        ele = - ele * x * x / 2.0 / i
        normal = normal + ele
        erf = erf + ele * x / (2.0 * i + 1)

    return np.clip(normal / np.sqrt(2.0 * np.pi) / sigma, 0, None), np.clip(erf / np.sqrt(2.0 * np.pi) / sigma, -0.5, 0.5)


#############################################################
# truncated_integral_and_sigma
# used by fitRobustGaussian function
def truncated_integral_and_sigma(x):
    n, e = normal_erf(x)
    return np.sqrt(1 - n * x / e)


#############################################################
# filterZscore
# Given a robustly fitted Gaussian parameters and an FPM threshold separating
# coverage associated with exon non-capture or capture during sequencing,
# exon are filtered when the Gaussian for the exon is indistinguishable from
# the non-capture threshold.
#
# Spec:
# - setting a tolerated deviation threshold, bdwthThreshold
# - check that the standard deviation is not == 0 otherwise no pseudo zscore
# can be calculated, change it if necessary
# - pseudo zscore calculation
# - comparison pseudo zscore with the tolerated deviation threshold => filtering
#
# Args:
# - mean [float], stdev [float]: parameters of the normal, requirement : mean > 0
# - unCaptFPMLimit [float]: FPM threshold separating captured and non-captured exons
#
# Returns "True" if exon doesn't pass the filter otherwise "False"
@numba.njit
def filterZscore(mean, stdev, unCaptFPMLimit):
    # Fixed paramater
    bdwthThreshold = 3  # tolerated deviation threshold
    meanDenom = 20

    if (mean == 0):
        raise Exception("filterZscore called with mean = 0.\n")

    if (stdev == 0):
        stdev = mean / meanDenom  # simulates 5% on each side of the mean

    zscore = (mean - unCaptFPMLimit) / stdev

    return zscore < bdwthThreshold


#############################################################
# filterSampsContrib2Gaussian
# Given a FPM counts from an exon and a robustly fitted Gaussian paramaters,
# filters the exons.
#
# Spec:
# - set a contribution threshold
# - obtain FPM values within +- 2 standard deviations of the mean of the Gaussian
# - calculate the contribution
# - compare the contribution to the threshold => filtering

# Args:
# - mean [float], stdev [float]: parameters of the normal
# - exFPMs (numpy.ndarray[floats]): FPM counts from an exon
#
# Returns "True" if exon doesn't pass the filter otherwise "False"
@numba.njit
def filterSampsContrib2Gaussian(mean, stdev, exFPMs):
    # Fixed parameters
    contribThreshold = 0.5
    stdevLim = 2

    FPMValuesUnderGaussian = exFPMs[(exFPMs > (mean - (stdevLim * stdev))) & (exFPMs < (mean + (stdevLim * stdev))), ]

    sampsContribution = len(FPMValuesUnderGaussian) / len(exFPMs)

    if (sampsContribution < contribThreshold):
        return True
    else:
        return False
