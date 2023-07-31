import logging
import os
import numba
import numpy as np
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time

import clusterSamps.smoothing
import CNCalls.CNCallsFile
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
#####################################
# clusterCalls
# Given a cluster ID, exon and intergenic fragment data, sample information,
# and various parameters.
# - Initialization: set up fixed parameters, define variables, and prepare output
#   file paths.
# - Retrieve cluster-specific information: Obtain the indices of target and control
#   samples associated with the cluster.
# - Preparation of data structures: allocate an array to store copy number (CN)
#   calls for each exon in the cluster and initialize relevant variables.
# -Cluster-specific analysis:
#   a. Exponential distribution fitting: fit an exponential distribution to the
#      intergenic fragment data of target and control samples within the cluster,
#      obtaining distribution parameters.
#      Generate a plot showing the profile of the exponential distribution.
#   b. Threshold calculation: calculate the threshold value in fragments per million (FPM)
#      corresponding to a specified fraction of the cumulative distribution function (CDF)
#      of the exponential distribution.
#   c. Exon processing: iterate over each exon in the cluster and analyze the copy
#      number data of target and control samples.
#       - Filtering: apply various filters to determine the filter status of the exon
#         based on its  copy number data and the obtained parameters.
#         If the exon fails any filter, it is marked accordingly and excluded from
#         further analysis.
#       - Plotting: if not in debug mode, generate a plot showing the profile of the
#         exon's exponential distribution.
# - Generate a pie chart summarizing the filter status counts.
#
# Args:
# - clustID [int]
# - exonsFPM (np.ndarray[floats]): normalised counts from exons
# - intergenicsFPM (np.ndarray[floats]): normalised counts from intergenic windows
# - exons (list of lists[str, int, int, str]): exon definitions [CHR,START,END,EXONID]
# - clusters (list of lists[ints]): each list corresponds to a cluster index containing the
#                                 associated "samples" indexes
# - ctrlsClusters (list of lists[int]): each list corresponds to a cluster index containing
#                                       control cluster index
# - specClusters (list[int]): each list index = cluster index, values = autosomes (0)
#                             or gonosomes (1)
# - exonsBool (np.ndarray[bool]): indicates if each exon is in a gonosomal region (True) or not (False).
# - outFile [str]: results calls file path
# - paramsToKeep (list[str])
#
# Returns a tuple (clustID, colInd4CNCallsArray, exonInd2Process, clusterCallsArray):
# - clustID [int]
# - colInd4CNCallsArray (list[ints]): column indexes for the CNcallsArray.
# - exonInd2Process (list[ints]): exons indexes (rows indexes) for the CNcallsArray.
# - clusterCallsArray (np.ndarray[floats]): The cluster calls array (likelihoods).
def clusterCalls(clustID, exonsFPM, intergenicsFPM, exons, clusters, ctrlsClusters,
                 specClusters, exonsBool, outFile, paramsToKeep):
    startTime = time.time()
    #############
    ### fixed parameters
    # fraction of the CDF of CNO exponential beyond which we truncate this distribution
    fracCDFExp = 0.99
    exonStatus = ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "exonsCalls"]
    # counter dictionary for each filter, only used to plot the pie chart
    exonStatusCountsDict = {status: 0 for status in exonStatus}
    # get output folder name
    outFolder = os.path.dirname(outFile)
    # defined file plot paths
    expFitPlotFile = "exponentialFit_coverageProfileSmoothed.pdf"
    clusterPieChartFile = "exonsFiltersSummary_pieChart.pdf"

    try:
        #############
        ### get cluster-specific informations
        try:
            (targetSampsInd, ctrlSampsInd) = getClusterSamps(clustID, ctrlsClusters, clusters)
        except Exception as e:
            logger.error("getClusterSamps failed for cluster %i : %s", clustID, repr(e))
            raise

        # To Fill and return
        exonInd2Process = np.where(exonsBool == specClusters[clustID])[0]
        clusterParamsArray = CNCalls.CNCallsFile.allocateParamsArray(len(exonInd2Process), 1, len(paramsToKeep))
        colInd4ParamsArray = [clustID * len(paramsToKeep), clustID * len(paramsToKeep) + 1]

        try:   # name of the output cluster folder and the list of subdirectories
            (clustFolder, pathPlotFolders) = makeResFolders(clustID, outFolder, exonStatus)
        except Exception as e:
            logger.error("makeResFolders failed for cluster %i : %s", clustID, repr(e))
            raise

        # DEBUG tracking
        logger.debug("cluster n°%i, clustSamps=%i, controlSamps=%i, exonsNB=%i",
                     clustID, len(targetSampsInd), len(ctrlSampsInd), len(exonInd2Process))

        ################
        ### Call process
        # fit an exponential distribution from all intergenic regions (CN0)
        try:
            (meanIntergenicFPM, loc, invLambda) = fitExponential(intergenicsFPM[:, targetSampsInd + ctrlSampsInd])
            expParams = {'distribution': scipy.stats.expon, 'loc': loc, 'scale': invLambda}
            preprocessExponFitPlot(clustID, meanIntergenicFPM, expParams, os.path.join(clustFolder, expFitPlotFile))
            clusterParamsArray[-1, 0] = loc
            clusterParamsArray[-1, 1] = invLambda
        except Exception as e:
            logger.error("fitExponential failed for cluster %i : %s", clustID, repr(e))
            raise

        # Calculating the threshold in FPM equivalent to 99% of the CDF (PPF = percent point function)
        unCaptFPMLimit = scipy.stats.expon.ppf(fracCDFExp, loc=loc, scale=invLambda)

        # Browse cluster-specific exons
        for ex in range(len(exonInd2Process)):
            ### exon-specific data
            exonInd4Exons = exonInd2Process[ex]
            exonInfos = exons[exonInd4Exons]  # list [CHR, START, END, EXONID]
            exonFPM4Clust = exonsFPM[exonInd4Exons, targetSampsInd + ctrlSampsInd]

            # dictionary list containing information about the fitted laws
            # {distribution:str, loc:float, scale:float}, list CNtype ordered, i.e. CN0, CN1, CN2, CN3.
            params = {}
            # retain parameters of law associated with homodeletions
            params["CN0"] = expParams

            try:
                filterStatus = exonCalls(ex, exonFPM4Clust, params, unCaptFPMLimit, clusterParamsArray)
            except Exception as e:
                logger.error("exonCalls failed : %s", repr(e))
                raise

            if filterStatus is not None:  # Exon filtered
                exonStatusCountsDict[filterStatus] += 1
                if pathPlotFolders:
                    try:
                        preprocessExonProfilePlot(filterStatus, exonStatusCountsDict, params,
                                                  unCaptFPMLimit, exonInfos, exonFPM4Clust, pathPlotFolders)
                    except Exception as e:
                        logger.error("preprocessExonProfilePlot failed : %s", repr(e))
                        raise

            else:  # Exon passed filters
                exonStatusCountsDict["exonsCalls"] += 1

        try:
            figures.plots.plotPieChart(clustID, exonStatusCountsDict, os.path.join(clustFolder, clusterPieChartFile))
        except Exception as e:
            logger.error("plotPieChart failed for cluster %i : %s", clustID, repr(e))
            raise

        # real time monitoring of function execution times
        for key, value in timer.timer_dict.items():
            logger.debug("ClustID: %s,  %s : %.2f sec", clustID, key, value)

        thisTime = time.time()
        logger.debug("ClusterCalls ClustID: %s execution time, in %.2fs", clustID, thisTime - startTime)
        return (clustID, colInd4ParamsArray, np.append(exonInd2Process, len(clusterParamsArray)), clusterParamsArray)

    except Exception as e:
        logger.error("CNCalls failed for cluster n°%i - %s", clustID, repr(e))
        raise Exception(str(clustID))


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################################################
# getClusterSamps
# Given a cluster ID, a list of control clusters for each cluster,
# and a list of sample indexes for each cluster.
# It returns a tuple containing the target sample indexes (specific to the cluster)
# and the combined sample indexes (including both target and control samples).
#
# Args:
# - clustID [int]: cluster identifier
# - ctrlsClustList (list of lists[int]]): control clusters for each cluster.
# - clusters (list of lists[int]]): sample indexes for each cluster.
#
# Returns a tuple (targetSampsInd, allSampsInd) containing:
#  - targetSampsInd [list[int]]: sample indexes specific to the target cluster.
#  - allSampsInd [list[int]]: combined sample indexes (target + control).
@timer.measure_time
def getClusterSamps(clustID, ctrlsClustList, clusters):
    targetSampsInd = clusters[clustID]
    ctrlSampsInd = []

    if len(ctrlsClustList[clustID]) != 0:
        # Extract cluster control sample indexes
        for ctrl_index in ctrlsClustList[clustID]:
            ctrlSampsInd.extend(clusters[ctrl_index])

    return (targetSampsInd, ctrlSampsInd)


############################################
# makeResFolders
# creates the cluster directory within the main plot folder.
# It then checks if the logging level is set to DEBUG.
# If it is, the function proceeds to create the specified subdirectories
# within the cluster directory based on the provided folderNames.
# The paths to these subdirectories are added to the pathPlotFolders list.
# Note: The function handles exceptions if directory creation fails and raises
# an exception with an appropriate error message.
#
# Args:
# - clusterID[int]: cluster identifier for which directories will be created.
# - outputFolder [str]: path to the main plot folder where the cluster directories should be created.
# - folderNames (list[str]): folder names to be created within each cluster directory.
#                            These folders are created only in DEBUG mode.
#
# Returns a tuple (clustFolder, pathPlotFolders):
# - clustFolder [str]: Path to the created cluster directory.
# - pathPlotFolders (list[str]): paths to the subdirectories created within the cluster directory.
@timer.measure_time
def makeResFolders(clusterID, outputFolder, folderNames):
    # To fill and return
    pathPlotFolders = []

    clusterFolder = os.path.join(outputFolder, "cluster_" + str(clusterID))
    try:
        os.makedirs(clusterFolder, exist_ok=True)
    except OSError:
        raise Exception("Error creating directory " + clusterFolder)

    if (logging.getLogger().getEffectiveLevel() <= logging.DEBUG):
        for folderIndex in range(len(folderNames)):
            folderPath = os.path.join(clusterFolder, folderNames[folderIndex])
            try:
                os.makedirs(folderPath, exist_ok=True)
            except OSError:
                raise Exception("Error creating directory " + folderPath)
            pathPlotFolders.append(folderPath)

    return (clusterFolder, pathPlotFolders)


#############################################################
# fitExponential
# Given a count array (dim = exonNB*samplesNB), fits an exponential distribution,
# setting location = 0.
#
# Args:
# - intergenicsFPMClust (np.ndarray[floats]): count array (FPM normalized)
#
# Returns a tuple (meanIntergenicFPM, loc, scale):
# - meanIntergenicFPM (np.ndarray[floats]): average count per exon, array needed for
#                                           the debug plot (preprocessExponFitPlot)
# - loc [float], scale[float]: parameters of the exponential
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
# Generates an exponential fit plot for a given cluster using the provided data and parameters.
# It performs data smoothing, calculates the probability density function for the exponential
# distribution, and plots the exponential fit.
#
# Args:
# - clustID [int]: The ID of the cluster.
# - meanIntergenicFPM (list[floats]): The list of mean intergenic FPM values.
# - params (list[floats]): loc and scale(invLambda) parameters of the exponential distribution.
# - file [str]: The path to the file where the plot should be saved.
#
# Return None
@timer.measure_time
def preprocessExponFitPlot(clustID, meanIntergenicFPM, params, file):
    # initiate plots variables
    plotTitle = f"ClusterID n° {clustID}"
    yLists = []  # List to store the y-values for plotting
    plotLegs = []  # List to store the plot legends

    # smoothing raw data
    # Smooth the average coverage profile of intergenic regions using the specified bandwidth
    try:
        (dr, dens, bwValue) = clusterSamps.smoothing.smoothData(meanIntergenicFPM, maxData=max(meanIntergenicFPM))
    except Exception as e:
        logger.error('smoothing failed for %s : %s', str(bwValue), repr(e))
        raise

    plotLegs.append("coverage data smoothed \nbwValue={:0.2e}".format(bwValue))
    yLists.append(dens)

    # calculate the probability density function for the exponential distribution
    yLists.append(computeLikelihood(params, dr))
    plotLegs.append("expon={:.2f}, {:.2f}".format(params['loc'], params['scale']))

    figures.plots.plotExponentialFit(plotTitle, dr, yLists, plotLegs, file)


################################################
# computeLikelihood
# Returns distributions's likelihood at the FPMs coordinates.
# Args:
# - params (dict): distribution parameters, including 'distribution':scipy.distributionName,
#                  'loc':float, and 'scale':float.
# - FPMs (list[floats])
#
# Returns a np.ndarray[floats]: probability density (likelihood) for a concerned distribution and
#                               FPMs.
@timer.measure_time
def computeLikelihood(params, FPMs):
    # Separate parts of parameters
    distribution = params['distribution']
    loc = params['loc']
    scale = params['scale']

    # scipy v1.5.4 default np.float64 cannot be changed
    y = distribution.pdf(FPMs, loc=loc, scale=scale)

    return y


##############################################
# exonCalls
# processes an exon by applying a series of filters.
# No specific output: It either returns a filter status string if the exon doesn't pass
# the filters or updates the clusterCallsArray with the computed likelihoods for the exon.
# Specs:
# 1) Check if the exon is not captured (median coverage = 0).
# 2) Fit a robust Gaussian distribution (CN2) to the exon FPM values.
# 3) Check if the fitted Gaussian overlaps the threshold associated with the uncaptured exon profile.
# 4) Check if the sample contribution rate to the Gaussian is too low (<50%).
# 5) If any of the above filters is triggered, return the corresponding filter status.
#
# Args:
# - exIndToProcess [int]
# - exonFPM4Clust (numpy.ndarray[floats]): Exon FPM (Fragments Per Million) values for the cluster.
# - params [dict]: parameters for the distribution models.
# - unCaptFPMLimit [float]: FPM threshold value for uncaptured exons.
# - clusterParamsArray (numpy.ndarray[floats]): store the Gaussian parameters [loc, scale].
@timer.measure_time
def exonCalls(exIndToProcess, exonFPM4Clust, params, unCaptFPMLimit, clusterParamsArray):
    #################
    ### init exon variable
    # a str which takes the name of the filter excluding the exon
    # (same name as the exFiltersReport keys). Remains "None" if exon is callable
    filterStatus = None

    ### Filter n°1: not captured => median coverage of the exon = 0
    if filterUncapturedExons(exonFPM4Clust):
        return "notCaptured"

    ### robustly fit a gaussian (CN2)
    ### Filter n°2: fitting is impossible (median close to 0)
    try:
        (mean, stdev) = fitRobustGaussian(exonFPM4Clust)
        # retain parameters of law associated with normal case (CN2)
        params["CN2"] = {'distribution': scipy.stats.norm, 'loc': mean, 'scale': stdev}
    except Exception as e:
        if str(e) == "cannot fit":
            return "cannotFitRG"
        else:
            raise Exception("fitRobustGaussian %s", repr(e))

    ### Filter n°3: fitted gaussian overlaps the threshold associated with the uncaptured exon profile
    if ((filterStatus is None) and (filterZscore(mean, stdev, unCaptFPMLimit))):
        return "RGClose2LowThreshold"

    ### Filter n°4: the samples contribution rate to the gaussian is too low (<50%)
    if ((filterStatus is None) and (filterSampsContrib2Gaussian(mean, stdev, exonFPM4Clust))):
        return "fewSampsInRG"

    clusterParamsArray[exIndToProcess, 0] = mean
    clusterParamsArray[exIndToProcess, 1] = stdev


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


#########################
# preprocessExonProfilePlot
# generates exon profile plots using given data, including probability density functions
# for different distributions.
# It checks conditions, prepares titles and variables, calculates distributions,
# creates plots, and saves them in a PDF file.
#
# Args:
# - status [str]: Status of the exon filters.
# - exonsFiltersSummary (dict): Summary of exon filters. keys = filter status[str], value= exons counter
# - params (list[dict]): List of parameter dictionaries for generating PDFs.
# - unCaptThreshold [float]: FPM threshold separating captured and non-captured exons
# - exonInfos (list[int]): Exon information [CHR,START,END,EXONID].
# - exonFPM (np.ndarray[floats]): Array of FPM values for one exon.
# - folders (list[str]): folder paths for storing the generated plots.
# - samples (list[str], optional): sample names. Defaults to None.
# - clusterSamps2Plot (list[int], optional): sample indexes to plot. Defaults to None.
# - sampsInd (list[int], optional): all sample indexes. Defaults to None.
@timer.measure_time
def preprocessExonProfilePlot(status, exonsFiltersSummary, params, unCaptThreshold,
                              exonInfos, exonFPM, folders):
    # Fixed parameter
    nbExToPlot = 500  # No. of threshold multiplier exons = one plot

    if status != "exonsCalls":
        if exonsFiltersSummary[status] % nbExToPlot != 0:
            return

    # Find the appropriate folder based on the 'status' value
    folder = next((f for f in folders if status in f), None)

    # Prepare the file and plot titles
    fileTitle = f"coverage_profil_{'_'.join(map(str, exonInfos))}"
    plotTitle = f"{'_'.join(map(str, exonInfos))}"
    # Lists to store plot data
    yLists = []
    plotLegs = [f"expon={params['CN0']['loc']:.2f}, {params['CN0']['scale']:.2f}"]
    verticalLines = [unCaptThreshold]
    vertLinesLegs = [f"UncoverThreshold={unCaptThreshold:.3f}"]
    # creates FPM ranges base
    xi = np.linspace(0, np.max(exonFPM), 1000)
    # calculate the probability density function for the exponential distribution
    yLists.append(computeLikelihood(params["CN0"], xi))
    ylim = np.max(yLists[0]) / 10

    if len(params) > 1:
        # calculate the probability density function for the gaussian distribution (CN2)
        yLists.append(computeLikelihood(params["CN2"], xi))
        plotLegs.append(f"RG CN2={params['CN2']['loc']:.2f}, {params['CN2']['scale']:.2f}")
        ylim = 2 * np.max(yLists[1])

    else:
        PDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, fileTitle))
        figures.plots.plotExonProfile(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_File)
        PDF_File.close()
