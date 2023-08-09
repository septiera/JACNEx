import logging
import os
import numba
import numpy as np
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time

import clusterSamps.smoothing
import figures.plots

# prevent PIL and numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


class Timer:
    # Initializes a Timer object.
    # Args:
    #     debug_mode (bool): Indicates whether the logger is in debug mode. Defaults to False
    def __init__(self, debug_mode=False):
        self.timer_dict = {}  # Initialize an empty dictionary
        self.debug_mode = debug_mode

    # Decorator function that measures the execution time of a function.
    # Args:
    #     func (callable): The function to measure the execution time of.
    # Returns:
    #     callable: The wrapped function.
    def measure_time(self, func):
        # Wrapper function that measures the execution time of the decorated function.
        # Args:
        #     *args: Variable-length argument list.
        #     **kwargs: Arbitrary keyword arguments.
        # Returns:
        #     The result of the decorated function.
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Record the start time
            result = func(*args, **kwargs)  # Execute the function
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time

            # Add the execution time to the dictionary
            if func.__name__ not in self.timer_dict:
                self.timer_dict[func.__name__] = 0
            self.timer_dict[func.__name__] += execution_time

            return result
        return wrapper


timer = Timer()


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
# CN0ParamsAndFPMLimit
# calculates the parameters "expon_loc" and "exp_scale" associated with the fitted exponential
# distribution from the intergenic count data.
# It also calculates the "unCaptFPMLimit," which is the FPM threshold equivalent to 99% of
# the cumulative distribution function (CDF) of the exponential distribution.
# Additionally, a PDF plot depicting the fitted exponential distribution is generated and saved
# in the specified 'plotDir'.
#
# Args:
# - intergenicsFPM (np.ndarray[floats]): count data from intergenic regions.
# - plotDir [str]: Path to the directory where plots will be saved.
#
# Returns:
# - expon_loc [float]: The location parameter of the fitted exponential distribution.
# - exp_scale [float]: The scale parameter of the fitted exponential distribution.
# - unCaptFPMLimit [float]: FPM threshold
def CN0ParamsAndFPMLimit(intergenicsFPM, plotDir):
    # fraction of the CDF of CNO exponential beyond which we truncate this distribution
    fracCDFExp = 0.99

    nbInterRegions = intergenicsFPM.shape[0]
    nbSamps = intergenicsFPM.shape[1]

    try:
        # Fit an exponential distribution to the intergenic fragment per million (FPM) data
        (meanIntergenicFPM, expon_loc, exp_scale) = fitExponential(intergenicsFPM)
    except Exception as e:
        logger.error("fitExponential failed : %s", repr(e))
        raise

    # Calculate the FPM limit below which exons are considered "uncaptured"
    unCaptFPMLimit = scipy.stats.expon.ppf(fracCDFExp, loc=expon_loc, scale=exp_scale)

    try:
        # Generate a plot to visualize the fitted exponential distribution
        nameFile = "exponentialFittedOn_" + str(nbSamps) + "Samps_" + str(nbInterRegions) + "intergenicRegions.pdf"
        pdfFile = os.path.join(plotDir, nameFile)
        preprocessExponFitPlot(meanIntergenicFPM, expon_loc, exp_scale, pdfFile)
    except Exception as e:
        logger.error("preprocessExponFitPlot failed : %s", repr(e))
        raise

    return (expon_loc, exp_scale, unCaptFPMLimit)


#####################################
# exonFilterAndCN2Params
# identifies and processes exons specific to a given cluster, along with its
# control clusters.
# It filters or calls the exons based on their fragment count data and stores
# the results (parameters of a Gaussian distribution and exon filtering status] in
# a cluster-specific array.
# Additionally, it generates a pie chart showing the status of the filtered
# or called exons for the cluster.
#
# Args:
# -clustID [str]
# -exonsFPM (np.ndarray[floats]): normalised counts from exons
# -samples (list[strs]): sample IDs in the same order as the columns of 'exonsFPM'.
# -clust2samps (dict): mapping cluster IDs to lists of sample IDs.
# -fitWith (dict): mapping cluster IDs to lists of control cluster IDs.
# -exonOnSexChr (numpy.ndarray[ints]): chromosome category (autosomes=0 or gonosomes chrXZ=1, chrYW=2)
#                                      for each exon.
# -unCaptFPMLimit (float): Upper limit for fragment count data, used in exon filtering.
# -expectedColNames (list[strs]): List of expected column names for the results array.
# -filterStatus (list[strs]): List of status labels used in filtering or calling exons.
#
# Returns a tuple (clustID, colInd4CNCallsArray, exonInd2Process, clusterCallsArray):
# - clustID [int]
# - colInd4CNCallsArray (list[ints]): CN2ParamsArray column indexes .
# - exonInd2Process (list[ints]): exons indexes (rows indexes) for the CNcallsArray.
# - clusterCallsArray (np.ndarray[floats]): The cluster calls array (likelihoods).
def exonFilterAndCN2Params(clustID, exonsFPM, samples, clust2samps, fitWith, exonOnSexChr,
                           unCaptFPMLimit, expectedColNames, filterStatus):
    startTime = time.time()
    try:
        #############
        ### get cluster-specific informations
        #############
        # Get the index of the current cluster in the list of keys from 'clust2samps'
        # enabling parallel storage of results in CN2ParamsArray
        clustInd = list(clust2samps).index(clustID)

        # Get target and control 'samples' indexes for each cluster
        # enables retrieval of fragment count data from exonsFPM array for all samples of interest,
        # including the target cluster and control clusters.
        try:
            (targetSampsInd, ctrlSampsInd) = getSampIndex(clustID, clust2samps, samples, fitWith)
        except Exception as e:
            logger.error("getClusterSamps failed for cluster %i : %s", clustID, repr(e))
            raise

        # Get appropriate chromosome category for clustID (autosomes or gonosomes)
        # enables identification of cluster-specific exons for analysis
        chrMap = {"A": 0, "XZ": 1, "YW": 2}
        clustChr = clustID.split("_")[0]
        # sanity check
        if clustChr not in chrMap:
            raise Exception("Cluster identifier not expected: ", clustID)

        #############
        # Data to be returned
        exonInd2Process = np.where(exonOnSexChr == chrMap[clustChr])[0]
        colInd4ParamsArray = np.arange(clustInd * len(expectedColNames), clustInd * len(expectedColNames) + len(expectedColNames))
        clusterParamsArray = np.full((len(exonInd2Process), len(expectedColNames)), -1, dtype=np.float64, order='F')

        #############
        # cluster summary
        logger.debug("cluster n°%s, clustSamps=%i, controlSamps=%i, exonsNB=%i",
                     clustID, len(targetSampsInd), len(ctrlSampsInd), len(exonInd2Process))

        #############
        ### Filter or call exons
        #############
        # Loop through each exon index to process
        for ei in range(len(exonInd2Process)):
            exonBedInd = exonInd2Process[ei]
            test = targetSampsInd + ctrlSampsInd
            exonFPMs = exonsFPM[exonBedInd, test]
            print(exonFPMs.shape)

            try:
                # Perform filtering or calling for the current exon and get the status
                exStatus = exonFilteredOrCalled(ei, exonFPMs, unCaptFPMLimit, clusterParamsArray, expectedColNames)
            except Exception as e:
                logger.error("exonFilteredOrCalled failed : %s", repr(e))
                raise

            # Update the clusterParamsArray with the status of the current exon
            clusterParamsArray[ei, expectedColNames.index("filterStatus")] = filterStatus.index(exStatus)

        # real-time monitoring of function execution times
        for key, value in timer.timer_dict.items():
            logger.debug("ClustID: %s,  %s : %.2f sec", clustID, key, value)

        logger.info(clusterParamsArray.shape)
        thisTime = time.time()
        logger.debug("ClusterCalls ClustID: %s execution time, in %.2fs", clustID, thisTime - startTime)
        return (clustID, colInd4ParamsArray, exonInd2Process, clusterParamsArray)

    except Exception as e:
        logger.error("CNCalls failed for cluster n°%i - %s", clustID, repr(e))
        raise Exception(str(clustID))


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################
# fitExponential
# Given a count array (dim = nbOfExons*nbOfSamps), fits an exponential distribution,
# setting location = 0.
#
# Args:
# - intergenicsFPM (np.ndarray[floats]): count array (FPM normalized)
#
# Returns a tuple (meanIntergenicFPM, loc, scale):
# - meanIntergenicFPM (np.ndarray[floats]): average count per exon, array needed for
#                                           'preprocessExponFitPlot' function
# - loc [float], scale[float]: parameters of the exponential distribution
@timer.measure_time
def fitExponential(intergenicsFPMClust):
    # compute meanFPM for each intergenic region (speed up)
    meanIntergenicFPM = np.mean(intergenicsFPMClust, axis=1)

    # Fit an exponential distribution, imposing location = 0
    # f(x, scale) = (1/scale)*exp(-x/scale)
    # scale = 1 / lambda
    loc, invLambda = scipy.stats.expon.fit(meanIntergenicFPM)

    return (meanIntergenicFPM, loc, invLambda)


###########################
# preprocessExponFitPlot
# Generates an exponential fit plot using the provided data and parameters.
# It performs data smoothing, calculates the probability density function for the exponential
# distribution, and plots the exponential fit.
#
# Args:
# - meanIntergenicFPM (list[floats]): The list of mean intergenic FPM values.
# - expon_loc[float], exp_scale[float]: parameters of the exponential distribution.
# - file [str]: The path to the file where the plot should be saved.
#
# Return None
@timer.measure_time
def preprocessExponFitPlot(meanIntergenicFPM, expon_loc, exp_scale, file):
    # initiate plots variables
    yLists = []  # List to store the y-values for plotting
    plotLegs = []  # List to store the plot legends

    # Smooth the average coverage profile of intergenic regions using the specified bandwidth
    try:
        (dr, dens, bwValue) = clusterSamps.smoothing.smoothData(meanIntergenicFPM, maxData=max(meanIntergenicFPM))
    except Exception as e:
        logger.error('smoothing failed for %s : %s', str(bwValue), repr(e))
        raise

    plotLegs.append("coverage data smoothed \nbwValue={:0.2e}".format(bwValue))
    yLists.append(dens)

    # calculate the probability density function for the exponential distribution
    yLists.append(scipy.stats.expon.pdf(dr, loc=expon_loc, scale=exp_scale))
    plotLegs.append("expon={:.2f}, {:.2f}".format(expon_loc, exp_scale))

    figures.plots.plotExponentialFit(dr, yLists, plotLegs, file)


#############################################################
# getSampleIndex
# identifies the indices of 'samples' belonging to the target cluster and control clusters.
# Control clusters consist of samples with coverage profiles similar to those in the target cluster.
# also checks for potential errors where a sample could be present in both a control and
# a target cluster, raising an error if such a case is detected.
#
# Args:
# -clustID [str]
# -clust2samps (dict): key==clusterID, value == list of sampleIDs
# -samples (list[strs]): sampleIDs, same order as the columns of FPMarray
# -fitWith(dict): key==clusterID, value == list of clusterIDs
#
# Returns a tuple (targetSampsInd, allSampsInd) containing two lists
# - the 'samples' indexes of target samples and control samples.
@timer.measure_time
def getSampIndex(clustID, clust2samps, samples, fitWith):
    # Convert the list of samples for the current cluster (clusterID)
    # into sets for faster membership checks
    targetSamps = set(clust2samps[clustID])
    ctrlSamps = set()

    # controlClusters is not empty, extract the control 'samples' names
    if len(fitWith[clustID]) != 0:
        for ctrlInd in fitWith[clustID]:
            ctrlSamps.update(clust2samps[ctrlInd])

    # Lists to store the 'samples' indexes of target and control samples
    targetSampsInd = []
    ctrlSampsInd = []

    for sampInd, sampName in enumerate(samples):
        # sanity check
        if sampName in targetSamps and sampName in ctrlSamps:
            logger.error('%s is present in both the control cluster %s and the target cluster %s',
                         sampName, clustID, " ".join(fitWith[clustID]))
            raise

        # Check if the current sample is in the target and/or control group
        isTargetSamp = sampName in targetSamps
        isCtrlSamp = sampName in ctrlSamps

        # Append the sample index to the appropriate list based on its group membership
        if isTargetSamp and not isCtrlSamp:
            targetSampsInd.append(sampInd)
        elif not isTargetSamp and isCtrlSamp:
            ctrlSampsInd.append(sampInd)

    return (targetSampsInd, ctrlSampsInd)


##############################################
# exonFilteredOrCalled
# Processes an exon by applying a series of filters to determine its filter status.
# Specs:
# 1) Check if the exon is not captured (median coverage = 0).
# 2) Fit a robust Gaussian distribution (CN2) to the exon FPM values.
# 3) Check if the fitted Gaussian overlaps the threshold associated with the uncaptured exon profile.
# 4) Check if the sample contribution rate to the Gaussian is too low (<50%).
# 5) If any of the above filters is triggered, return the corresponding filter status.
# 6) If the exon passes all filters, the clusterParamsArray is updated with the computed parameters
# of the Gaussian distribution for the exon.
#
# Args:
# -exIndToProcess [int]
# -exonFPM (numpy.ndarray[floats]): Exon FPM (Fragments Per Million) values for the cluster.
# -unCaptFPMLimit [float]: FPM threshold associated with the uncaptured exons.
# -clusterParamsArray (numpy.ndarray[floats]): store the Gaussian parameters [loc, scale].
# -expectedColNames (list[strs]): List of column names for the clusterParamsArray ["loc","scale","filterStatus"]
# Returns:
# - str: exon filter status("notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", or "call").
# If the filter status is "call", it means the exon passes all filters and is callable.
# If the filter status is any other value, the exon does not pass one or more filters.
@timer.measure_time
def exonFilteredOrCalled(exIndToProcess, exonFPMs, unCaptFPMLimit, clusterParamsArray, expectedColNames):
    #################
    ### init exon variable
    # a str which takes the name of the filter excluding the exon
    # (same name as the exFiltersReport keys). Remains "None" if exon is callable
    filterStatus = None

    ### Filter n°1: not captured => median coverage of the exon = 0
    if filterUncapturedExons(exonFPMs):
        return "notCaptured"

    ### robustly fit a gaussian (CN2)
    ### Filter n°2: fitting is impossible (median close to 0)
    try:
        (gaussian_loc, gaussian_scale) = fitRobustGaussian(exonFPMs)

    except Exception as e:
        if str(e) == "cannot fit":
            return "cannotFitRG"
        else:
            raise Exception("fitRobustGaussian %s", repr(e))

    ### Filter n°3: fitted gaussian overlaps the threshold associated with the uncaptured exon profile
    if ((filterStatus is None) and (filterZscore(gaussian_loc, gaussian_scale, unCaptFPMLimit))):
        return "RGClose2LowThreshold"

    ### Filter n°4: the samples contribution rate to the gaussian is too low (<50%)
    if ((filterStatus is None) and (filterSampsContrib2Gaussian(gaussian_loc, gaussian_scale, exonFPMs))):
        return "fewSampsInRG"

    clusterParamsArray[exIndToProcess, expectedColNames.index("loc")] = gaussian_loc
    clusterParamsArray[exIndToProcess, expectedColNames.index("scale")] = gaussian_scale
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
# - exonFPM (list[floats]): FPM counts from an exon
#
# Returns "True" if exon doesn't pass the filter otherwise "False"
@timer.measure_time
@numba.njit
def filterUncapturedExons(exonFPM):
    medianFPM = np.median(exonFPM)
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
@timer.measure_time
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
@timer.measure_time
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
# - exonFPM (list[floats]): FPM counts from an exon
#
# Returns "True" if exon doesn't pass the filter otherwise "False"
@timer.measure_time
@numba.njit
def filterSampsContrib2Gaussian(mean, stdev, exonFPM):
    # Fixed parameters
    contribThreshold = 0.5
    stdevLim = 2

    FPMValuesUnderGaussian = exonFPM[(exonFPM > (mean - (stdevLim * stdev))) & (exonFPM < (mean + (stdevLim * stdev))), ]

    sampsContribution = len(FPMValuesUnderGaussian) / len(exonFPM)

    if (sampsContribution < contribThreshold):
        return True
    else:
        return False
