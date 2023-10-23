import logging
import numba
import numpy as np
import scipy.stats

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
# computeCN0Params
# Calculates the parameters "expon_loc" and "exp_scale" associated with the fitted exponential
# distribution from the intergenic count data. It also calculates the "uncaptThreshold".
# It is the FPM (Fragments Per Million) limit below which exons are considered "uncaptured"
# due to insufficient sequencing or capture. It's derived from the intergenic fragment distribution
# and set at the 99th percentile of the associated exponential distribution (CN0).
# Exons with FPM values below the threshold are marked as "uncaptured."
#
# Args:
# - intergenicsFPM (np.ndarray[floats]): count data from intergenic regions.
#
# Returns:
# - expon_loc [float]: location parameter of the fitted exponential distribution.
# - expon_scale [float]: scale parameter of the fitted exponential distribution.
# - meanIntergenicFPM (np.ndarray[floats]): average fragment counts per region,
#                                           array needed for debug/test script
# - uncaptThreshold [float]: FPM threshold
def computeCN0Params(intergenicsFPM):
    # Fraction of the CDF of CNO exponential beyond which we truncate this distribution
    fracCDFExp = 0.99
    try:
        # Fit an exponential distribution to the intergenic fragment per million (FPM) data
        (expon_loc, expon_scale, meanIntergenicFPM) = fitExponential(intergenicsFPM)
    except Exception as e:
        logger.error("fitExponential failed : %s", repr(e))
        raise

    # Calculate the FPM limit below which exons are considered "uncaptured"
    uncaptThreshold = scipy.stats.expon.ppf(fracCDFExp, loc=expon_loc, scale=expon_scale)

    return (expon_loc, expon_scale, meanIntergenicFPM, uncaptThreshold)


#####################################
# processExonsAndComputeCN2Params
# Processes exon data, applies filters, and computes CN2 distribution parameters
# for a cluster.
# Metrics, including Gaussian distribution parameters and filter states,
# are stored in a cluster-specific array.
#
# Args:
# - clusterID [str]
# - chromType [str]: "A" for autosomes, "G" for gonosomes
# - exonFPMs (np.ndarray[floats]): normalised counts from exons
# - samples (list[strs]): sample IDs in the same order as the columns of 'exonFPMs'.
# - clust2samps (dict): mapping cluster IDs to lists of sample IDs.
# - fitWith (dict): mapping cluster IDs to lists of control cluster IDs.
# - unCaptFPMLimit [float]: Upper limit for fragment count data, used in exon filtering.
# - metricsNames (list[strs]): List of expected column names for the results array.
# - filterStates (list[strs]): List of states labels used in filtering or calling exons.
#
# Returns a tuple (chromType, clusterID, clustExMetrics):
# - chromType [str]
# - clusterID [str]
# - clustExMetrics (dict): key == clusterID, value == np.ndarray[floats]
#                          dim = NbOfExons * NbOfExonsMetrics
#
# Raises an exception: If an error occurs during execution.
def processExonsAndComputeCN2Params(clusterID, chromType, exonFPMs, samples, clust2samps,
                                    fitWith, unCaptFPMLimit, metricsNames, filterStates):
    try:
        # Initialize an array to store metrics with -1 as default value
        clustExMetrics = np.full((exonFPMs.shape[0], len(metricsNames)), -1, dtype=np.float64, order='C')

        try:
            # Get sample indexes for the current cluster and it's fitWith cluster(s)
            sampsInd = getSampIndexes(clusterID, clust2samps, samples, fitWith)
        except Exception as e:
            logger.error("getSampIndexes failed for cluster %i : %s", clusterID, repr(e))
            raise

        logger.debug("cluster %s, nbSamps=%i", clusterID, len(sampsInd))

        # Iterate through each exon
        for ei in range(exonFPMs.shape[0]):
            # Extract exon FPMs for target samples
            exFPMs = exonFPMs[ei, sampsInd]

            try:
                # Determine filter applied and update clustExMetrics
                filter = exonFilteredOrCalled(ei, exFPMs, unCaptFPMLimit, clustExMetrics, metricsNames)
            except Exception as e:
                logger.error("exonFilteredOrCalled failed for exon index %i: %s", ei, repr(e))
                raise

            # Update filter state in metrics array
            clustExMetrics[ei, metricsNames.index("filterStates")] = filterStates.index(filter)

        return (chromType, clusterID, clustExMetrics)

    except Exception as e:
        logger.error("CNCalls failed for cluster %s - %s", clusterID, repr(e))
        raise Exception(str(clusterID))


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################
# fitExponential
# Given a fragment count array (dim = NBOfGenomicWindows*NBOfSamps),
# fits an exponential distribution, fixing location to 0.
#
# Args:
# - intergenicsFPM (np.ndarray[floats]): count array (FPM normalized)
#
# Returns a tuple (meanIntergenicFPM, loc, scale):
# - meanIntergenicFPM (np.ndarray[floats]): average count per exon, array needed for
#                                           debug/test script
# - expon_loc [float], expon_scale[float]: parameters of the exponential distribution
def fitExponential(intergenicsFPM):
    # compute meanFPM for each intergenic region (speed up)
    meanIntergenicFPM = np.mean(intergenicsFPM, axis=1)

    # Fit an exponential distribution,
    # enforces the "loc" parameter to be 0 because our model requires the distribution
    # to start at zero.
    # f(x, scale) = (1/scale)*exp(-x/scale)
    # scale = 1 / lambda
    expon_loc, expon_scale = scipy.stats.expon.fit(meanIntergenicFPM, floc=0)

    return (expon_loc, expon_scale, meanIntergenicFPM)


#############################################################
# getSampIndexes
# identifies the 'samples' indexes belonging to the current cluster and 'fitWith' cluster(s).
#
# Args:
# -clusterID [str]
# -clust2samps (dict): key==clusterID, value == list of sampleIDs
# -samples (list[strs]): sampleIDs, same order as the columns of "counts" array
# -fitWith(dict): key==clusterID, value == list of clusterIDs
#
# Returns a list of samples indexes from 'counts'
def getSampIndexes(clusterID, clust2samps, samples, fitWith):
    clustSamps = clust2samps[clusterID]
    fitWithSamps = set()

    for clustID in fitWith[clusterID]:
        fitWithSamps.update(clust2samps[clustID])

    clustSampsInd = [samples.index(sample) for sample in clustSamps if sample in samples]
    fitWithSampsInd = [samples.index(sample) for sample in fitWithSamps if sample in samples]

    return clustSampsInd + fitWithSampsInd


##############################################
# exonFilteredOrCalled
# Processes an exon by applying a series of filters to determine its filter status.
# Specs:
# 1) Check if the exon is not captured (median coverage = 0).
# 2) Fit a robust Gaussian distribution (CN2) to the exon FPM values.
# 3) Check if the fitted Gaussian overlaps the threshold associated with the uncaptured exon profile.
# 4) Check if the sample contribution rate to the Gaussian is too low (<50%).
# 5) If any of the above filters is triggered, return the corresponding filter status.
# 6) If the exon passes 2 first filters, the clusterParamsArray is updated with the computed parameters
#    of the Gaussian distribution for the exon.
#
# Args:
# -exIndToProcess [int]
# -exFPMs (numpy.ndarray[floats]): Exon FPM (Fragments Per Million) values for the cluster.
# -unCaptFPMLimit [float]: FPM threshold associated with the uncaptured exons.
# -clusterParamsArray (numpy.ndarray[floats]): store the Gaussian parameters [loc, scale].
# -expectedColNames (list[strs]): List of column names for the clusterParamsArray ["loc","scale","filterStatus"]
#
# Returns:
# - str: exon filter status("notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", or "call").
# If the filter status is "call", it means the exon passes all filters and is callable.
# If the filter status is any other value, the exon does not pass one or more filters.
def exonFilteredOrCalled(exIndToProcess, exFPMs, unCaptFPMLimit, clusterResArray, clustResColnames):
    #################
    ### init exon variable
    # a str which takes the name of the filter excluding the exon
    # (same name as the filterStatus strs).
    filterStatus = None

    ### Filter n°1: not captured => median coverage of the exon = 0
    if filterUncapturedExons(exFPMs):
        return "notCaptured"

    ### robustly fit a gaussian (CN2)
    ### Filter n°2: fitting is impossible (median close to 0)
    try:
        (gaussian_loc, gaussian_scale) = fitRobustGaussian(exFPMs)

    except Exception as e:
        if str(e) == "cannot fit":
            return "cannotFitRG"
        else:
            raise Exception("fitRobustGaussian %s", repr(e))

    clusterResArray[exIndToProcess, clustResColnames.index("loc")] = gaussian_loc
    clusterResArray[exIndToProcess, clustResColnames.index("scale")] = gaussian_scale

    ### Filter n°3: fitted gaussian overlaps the threshold associated with the uncaptured exon profile
    if ((filterStatus is None) and (filterZscore(gaussian_loc, gaussian_scale, unCaptFPMLimit))):
        return "RGClose2LowThreshold"

    ### Filter n°4: the samples contribution rate to the gaussian is too low (<50%)
    if ((filterStatus is None) and (filterSampsContrib2Gaussian(gaussian_loc, gaussian_scale, exFPMs))):
        return "fewSampsInRG"

    return "call"


###################
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
# Fits a single principal gaussian component around a starting guess point
# in a 1-dimensional gaussian mixture of unknown components with EM algorithm
# script found to :https://github.com/hmiemad/robust_Gaussian_fit (v01_2023)
#
# Args:
# - X (np.array): A sample of 1-dimensional mixture of gaussian random variables
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
        measure the standard deviation of the window and divide by the standard deviation of a truncated gaussian distribution
        measure the proportion of points inside the window, divide by the weight of a truncated gaussian distribution
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


###################
# filterZscore
# Given a robustly fitted gaussian parameters and an FPM threshold separating coverage
# associated with exon non-capture or capture during sequencing, exon are filtered when
# the gaussian  for the exon is indistinguishable from
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


###################
# filterSampsContrib2Gaussian
# Given a FPM counts from an exon and a robustly fitted gaussian paramaters,
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
