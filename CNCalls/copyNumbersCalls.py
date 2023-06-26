import logging
import os
import numba
import numpy as np
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time
import traceback

import clusterSamps.smoothing
import CNCalls.CNCallsFile
import figures.plots

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

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
# - samples (list [str]): samples names
# - exons (list of lists[str, int, int, str]): exon definitions [CHR,START,END,EXONID]
# - clusters (list of lists[ints]): each list corresponds to a cluster index containing the
#                                 associated "samples" indexes
# - ctrlsClusters (list of lists[int]): each list corresponds to a cluster index containing
#                                       control cluster index
# - specClusters (list[int]): each list index = cluster index, values = autosomes (0)
#                             or gonosomes (1)
# - CNTypes (list[str]): Copy number states.
# - exonsBool (np.ndarray[bool]): indicates if each exon is in a gonosomal region (True) or not (False).
# - outFile [str]: results calls file path
# - sampsExons2Check (list of lists[int]): exon indices for each "sample" index to be graphically
#                                          validated.
#
# Returns a tuple (clustID, colInd4CNCallsArray, exonInd2Process, clusterCallsArray):
# - clustID [int]
# - colInd4CNCallsArray (list[ints]): column indexes for the CNcallsArray.
# - exonInd2Process (list[ints]): exons indexes (rows indexes) for the CNcallsArray.
# - clusterCallsArray (np.ndarray[floats]): The cluster calls array.
def clusterCalls(clustID, exonsFPM, intergenicsFPM, samples, exons, clusters, ctrlsClusters,
                 specClusters, cnTypes, exonsBool, outFile, sampsExons2Check):
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
        clusterCallsArray = CNCalls.CNCallsFile.allocateCNCallsArray(len(exonInd2Process),
                                                                     len(targetSampsInd),
                                                                     len(cnTypes))
        colInd4CNCallsArray = [idx * len(cnTypes) + i for idx in targetSampsInd for i in range(len(cnTypes))]

        try:  # list of samples (from current cluster) and exons to be checked
            (sampsToCheck, exonsToCheck) = getClustInfo2Check(targetSampsInd, sampsExons2Check)
        except Exception as e:
            logger.error("getClustInfo2Check failed for cluster %i : %s", clustID, repr(e))
            raise

        try:   # name of the output cluster folder and the list of subdirectories
            (clustFolder, pathPlotFolders) = makeResFolders(clustID, outFolder, exonStatus, sampsToCheck)
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

            filterStatus = exonCalls(ex, exonFPM4Clust, params, unCaptFPMLimit, cnTypes,
                                     len(targetSampsInd), clusterCallsArray)

            if filterStatus is not None:  # Exon filtered
                exonStatusCountsDict[filterStatus] += 1
                if pathPlotFolders:
                    try:
                        preprocessExonProfilePlot(filterStatus, exonStatusCountsDict, params,
                                                  unCaptFPMLimit, exonInfos, exonFPM4Clust, pathPlotFolders)
                    except Exception as e:
                        logger.error("preprocessExonProfilePlot failed : %s", repr(e))
                        print(traceback.format_exc())
                        raise
                    
                if exonInd4Exons in exonsToCheck:
                    logger.error("cluster n°%s exonID %s to check according to the user is filtered",clustID,
                                 f"{'_'.join(map(str, exonInfos))}")
                continue

            else:  # Exon passed filters
                exonStatusCountsDict["exonsCalls"] += 1

                # plot exonCalls only in DEBUG mode
                if (pathPlotFolders) and (exonInd4Exons in exonsToCheck):
                    try:
                        preprocessExonProfilePlot("exonsCalls", exonStatusCountsDict, params,
                                                  unCaptFPMLimit, exonInfos, exonFPM4Clust, pathPlotFolders,
                                                  samples, sampsToCheck, targetSampsInd + ctrlSampsInd)

                    except Exception as e:
                        logger.error("preprocessExonProfilePlot2 failed : %s", repr(e))
                        raise

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
        return (clustID, colInd4CNCallsArray, exonInd2Process, clusterCallsArray)

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


#################################
# getClustInfo2Check
# Given a list of sample indexes (allSampsInd) representing the samples
# in the cluster, and a list of lists of exon indexes (sampsExons2Check) indicating
# the exons to be analyzed for each sample index.
# Identifies the samples in a cluster that need to be checked and the unique exons to be analyzed.
# Note that all the samples to be checked, even if they do not have the same exons to be checked,
# will be represented for all the exons. These samples are used as controls.
#
# Args:
# - allSampsInd (list[int]): Indexes of "samples" in the cluster.
# - sampsExons2Check (list of lists[int]): exon indices for each "sample" index to be graphically
#   validated.
#
# Returns:
# A tuple containing:
# - sampsToCheck (list[int]): Indexes of samples in the cluster that need to be checked.
# - exonsToCheck (list[int]): Indices of exons to analyze within the cluster.
@timer.measure_time
def getClustInfo2Check(allSampsInd, sampsExons2Check):
    sampsToCheck = []
    exonsToCheck = set()

    for sampInd in allSampsInd:
        if len(sampsExons2Check[sampInd]) > 0:  # sample has exons to be checked
            sampsToCheck.append(sampInd)
            # Update the set of exon indexes with the unique indexes from the sample
            exonsToCheck.update(sampsExons2Check[sampInd])

    return (sampsToCheck, list(exonsToCheck))


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
# - samplesToCheck (list[str]): sample names associated with the cluster
#
# Returns a tuple (clustFolder, pathPlotFolders):
# - clustFolder [str]: Path to the created cluster directory.
# - pathPlotFolders (list[str]): paths to the subdirectories created within the cluster directory.
@timer.measure_time
def makeResFolders(clusterID, outputFolder, folderNames, samplesToCheck):
    # To fill and return
    pathPlotFolders = []

    clusterFolder = os.path.join(outputFolder, "cluster_" + str(clusterID))
    try:
        os.makedirs(clusterFolder, exist_ok=True)
    except OSError:
        raise Exception("Error creating directory " + clusterFolder)

    if (logging.getLogger().getEffectiveLevel() <= logging.DEBUG):
        numFolders = len(folderNames)  # Store the length of folderNames
        for folderIndex in range(numFolders - 1):  # Exclude the last folder
            folderPath = os.path.join(clusterFolder, folderNames[folderIndex])
            try:
                os.makedirs(folderPath, exist_ok=True)
            except OSError:
                raise Exception("Error creating directory " + folderPath)
            pathPlotFolders.append(folderPath)

        # Check if the last folder should be created
        lastFolderIndex = numFolders - 1
        if samplesToCheck or (lastFolderIndex >= 0 and len(samplesToCheck) > 0):
            lastFolderPath = os.path.join(clusterFolder, folderNames[lastFolderIndex])
            try:
                os.makedirs(lastFolderPath, exist_ok=True)
            except OSError:
                raise Exception("Error creating directory " + lastFolderPath)
            pathPlotFolders.append(lastFolderPath)

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
    makePDF(params, dr, yLists)
    plotLegs.append("expon={:.2f}, {:.2f}".format(params['loc'], params['scale']))

    figures.plots.plotExponentialFit(plotTitle, dr, yLists, plotLegs, file)


################################################
# makePDF
# Generate distributions's Probability Distribution Function (likelihood) and added it to yLists
# Args:
# - params (dict[str:floats]): distribution parameters, including 'distribution', 'loc', and 'scale'.
# - rangeData (list[floats]): range of FPM data
# - yLists (list[floats]): computed PDF values are appended
@timer.measure_time
def makePDF(params, rangeData, yLists):
    # Separate parts of parameters
    distribution = params['distribution']
    loc = params['loc']
    scale = params['scale']

    # Build PDF
    x = rangeData
    y = distribution.pdf(x, loc=loc, scale=scale)
    yLists.append(y)


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
# 6) If the exon passes all filters, compute the likelihoods using the fitted parameters
# for heterodeletion and duplication (CN1 and CN3).
# 7) Store the computed likelihoods in the clusterCallsArray for the given exon index (exIndToProcess).
#
# Args:
# - exIndToProcess [int]
# - exonFPM4Clust (numpy.ndarray[floats]): Exon FPM (Fragments Per Million) values for the cluster.
# - params [dict]: parameters for the distribution models.
# - unCaptFPMLimit [float]: FPM threshold value for uncaptured exons.
# - cnTypes [list]: Types of copy number states.
# - nbTargetSamps [int]: Number of target samples.
# - clusterCallsArray (numpy.ndarray[floats]): store the cluster calls (likelihoods).
@timer.measure_time
def exonCalls(exIndToProcess, exonFPM4Clust, params, unCaptFPMLimit, cnTypes, nbTargetSamps, clusterCallsArray):
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

    # exon passed filters
    # retain parameters of law associated with heterodeletion and duplication
    params["CN1"] = {'distribution': scipy.stats.norm, 'loc': mean / 2, 'scale': stdev}
    params["CN3"] = {'distribution': scipy.stats.norm, 'loc': 3 * (mean / 2), 'scale': stdev}
    try:
        likelihoods = sampFPM2Probs(exonFPM4Clust, nbTargetSamps, params, cnTypes)
    except Exception as e:
        logger.error("sampFPM2Probs failed : %s", repr(e))
        raise
    # likelihoods array dim = distribsNB * samplesNB
    # flatten(order='F') specifies the Fortran-style (column-major)
    # order of flattening the array.
    clusterCallsArray[exIndToProcess] = likelihoods.flatten(order='F')


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


#############################################################
# sampFPM2Probs
# Converts exon FPM values into PDF probabilities (likelihoods)
# for a given set of parameters and sample indexes.
#
# Args:
# - exonFPM (np.ndarray[float]): Array of FPM values for one exon.
# - NbTargetSamps (int): Number of target samples associated with the cluster.
# - params (List[Dict]): List of parameter dictionaries for generating PDFs.
# - CNTypes (List[str]): List of CN types.
#
# Returns:
# -likelihoods (np.ndarray[float]): PDF values for each set of parameters and sample indexes.
#                                   Dimensions: (CN0, CN1, CN2, CN3+) * NbTargetSamps
@timer.measure_time
def sampFPM2Probs(exonFPM, NbTargetSamps, params, CNTypes):
    exonFPM4Clust = exonFPM[:NbTargetSamps]
    pdf_groups = []

    # Calculate the PDF for each set of parameters and each value of x
    for cn in CNTypes:
        makePDF(params[cn], exonFPM4Clust, pdf_groups)
    likelihoods = np.array(pdf_groups)

    return likelihoods


#############################################################
# fillClusterCallsArray
#
# Args:
# - clusterCalls (np.ndarray[floats]): copy number call likelihood
# - sampIndex [int]: "samples" index
# - exIndex [int]: exon index in clusterCalls array
# - likelihoods (list[floats]): likelihood for a sample and an exon
@timer.measure_time
def fillClusterCallsArray(clusterCalls, sampIndex, exIndex, likelihoods):
    sampIndInArray = sampIndex * len(likelihoods)
    # sanity check : don't overwrite data
    if np.sum(clusterCalls[exIndex, sampIndInArray:(sampIndInArray + len(likelihoods))]) <= 0:
        # Populate the clusterCalls with the copy number call likelihood probabilities
        clusterCalls[exIndex, sampIndInArray:(sampIndInArray + len(likelihoods))] = likelihoods
    else:
        raise Exception("overwrite calls in clusterCalls array")


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
                              exonInfos, exonFPM, folders, samples=None,
                              clusterSamps2Plot=None, sampsInd=None):
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
    makePDF(params["CN0"], xi, yLists)
    ylim = np.max(yLists[0]) / 10

    if len(params) > 1:
        # calculate the probability density function for the gaussian distribution (CN2)
        makePDF(params["CN2"], xi, yLists)
        plotLegs.append(f"RG CN2={params['CN2']['loc']:.2f}, {params['CN2']['scale']:.2f}")
        ylim = 2 * np.max(yLists[1])

    if clusterSamps2Plot is not None:
        # calculate the probability density function for the gaussian distribution (CN1)
        makePDF(params["CN1"], xi, yLists)
        plotLegs.append(f"RG CN1={params['CN1']['loc']:.2f}, {params['CN1']['scale']:.2f}")

        # calculate the probability density function for the gaussian distribution (CN3)
        makePDF(params["CN3"], xi, yLists)
        plotLegs.append(f"RG CN3={params['CN3']['loc']:.2f}, {params['CN3']['scale']:.2f}")
        
        for samp in clusterSamps2Plot:
            sampName = samples[samp]
            sampFPM = exonFPM[sampsInd.index(samp)]            

            # Create individual data for the current sample
            individualVerticalLines = [sampFPM]
            individualVertLinesLegs = [f"sample name={sampName}"]
            individualFileTitle = f"{sampName}_{fileTitle}"
            
            # Generate individual plot for the current sample
            individualPDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, individualFileTitle))
            figures.plots.plotExonProfile(exonFPM, xi, yLists, plotLegs, individualVerticalLines, individualVertLinesLegs, plotTitle, ylim, individualPDF_File)
            individualPDF_File.close()

    else:
        PDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, fileTitle))
        figures.plots.plotExonProfile(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_File)
        PDF_File.close()
