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

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)

class Timer:
    def __init__(self):
        self.timer_dict = {}  # Initialize an empty dictionary

    def measure_time(self, func):  # Decorator function that measures the execution time of a function
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Record the start time
            result = func(*args, **kwargs)  # Execute the function
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time

            # Add the execution time to the dictionary
            if func.__name__ not in self.timer_dict:
                self.timer_dict[func.__name__] = 0
            self.timer_dict[func.__name__] += execution_time

            return result  # Return the result of the function
        return wrapper  # Return the wrapped function
timer = Timer()


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#####################################
# CNCalls
# Given a cluster identifier, two arrays containing FPM values for intergenic regions and exons,
# as well as lists of definitions for samples and clusters, this function calculates the
# probabilities (likelihood) for each copy number type (CN0, CN1, CN2, CN3+) for each exon
# and each sample.
#
# Specifically:
# -Retrieves cluster-specific data such as exon and intergenic FPM (fragments per million)
# count data
# -Fits an exponential distribution from all intergenic FPM data to obtain the distribution parameters
# and an FPM threshold to identify captured and uncaptured profiles (unCaptFPMLimit).
# -Filters out exons that are not interpretable and fits a robust Gaussian:
#   -F1: filters exons with median coverage of 0, i.e., not captured.
#   -F2: filters exons where it is impossible to fit a robust Gaussian, i.e., median coverage
#        is close to 0.
#   -F3: filters exons where the fitted Gaussian overlaps the threshold associated with the
#        uncaptured exon profile.
#   -F4: filters exons where the sample contribution rate to the robust Gaussian is too low,
#        i.e., less than 50%.
# -Calculates likelihoods for each sample/exon to determine CN0 (case where the sample FPM
# value is lower than the CN1 mean, otherwise 0), CN1, CN2, and CN3 and fills clusterCalls
# -Produces filtered and unfiltered exon profiles representation (fit plot, pie chart) if a path
# for plotted is defined by the user, saved in different folders and PDF files.
#
# Args:
# - clustID [int]
# - exonsFPM (np.ndarray[float]): normalised counts from exons
# - intergenicsFPM (np.ndarray[float]): normalised counts from intergenic windows
# - samples (list [str]): samples names
# - exons (list of lists[str, int, int, str]): exon definitions [CHR,START,END,EXONID]
# - clusters (list of lists[ints]): each list corresponds to a cluster index containing the
#                                   associated "samples" indexes
# - ctrlsClusters (list of lists[int]): each list corresponds to a cluster index containing
#                                       control cluster index
# - sourceClusters (list[int]): each list index = cluster index, values = autosomes (0) or gonosomes (1)
# - maskSourceExons (np.ndarray[bool]): indicates if each exon is in a gonosomal region (True) or not (False)
# - plotFolder (str): subdir (created if needed) where result plots files will be produced
# - samps2Check (str): sample names for the graphic control of the number of copies called
#
# Returns a tuple (colIndInCNCallsArray, sourceExons, clusterCalls):
# - clustID [int]
# - colIndInCNCallsArray (list[int]): Column "samples" indexes in CNcallsArray for the current cluster.
# - sourceExons (list[int]): source exons indexes (gonosomes or autosomes exons) for the current cluster.
# - clusterCalls (np.ndarray): samples likelihoods array for the current cluster.
def CNCalls(clustID, exonsFPM, intergenicsFPM, samples, exons, clusters, ctrlsClusters,
            sourceClusters, maskSourceExons, plotFolder, samps2Check):
    # fixed parameter
    # fraction of the CDF of CNO exponential beyond which we truncate this distribution 
    fracCDFExp = 0.99

    try:
        #############
        ### initialising variables and extracting information from the cluster
        ### Fixed parameters
        probsStates = ["CNO", "CN1", "CN2", "CN3"]
        # counter dictionary for each filter, only used to plot the pie chart
        exonsFiltersSummary = dict.fromkeys(["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "exonsCalls"], 0)

        ### extracting cluster-specific informations
        clusterSampsIndexes = clusters[clustID].copy()
        clusterSampsNames = [samples[i] for i in clusterSampsIndexes]
        # cluster source 'exons' list (gonosomes or autosomes exons), output
        sourceExons = np.where(maskSourceExons == sourceClusters[clustID])[0]
        colIndInCNCallsArray = [idx * len(probsStates) + i for idx in clusterSampsIndexes for i in range(len(probsStates))]

        ### To Fill and return
        clusterCalls = np.full((len(sourceExons), (len(clusterSampsIndexes) * len(probsStates))), -1, dtype=np.float32, order='F')

        #### creation of folders for storing DEBUG plots
        if plotFolder:
            try:
                plotFolders = makePlotFolders(plotFolder, clustID, clusterSampsNames, samps2Check)
            except Exception as e:
                logger.error("makePlotFolders failed for cluster %i : %s", clustID, repr(e))
                raise

        # add sample indexes if there are control clusters
        if len(ctrlsClusters[clustID]) != 0:
            ctrlClusterSampsInd = getCTRLSamps(ctrlsClusters[clustID], clusters)
            sampsInd = clusterSampsIndexes + ctrlClusterSampsInd
        else:
            sampsInd = clusterSampsIndexes

        # DEBUG tracking
        logger.info("cluster n°%i, clustSamps=%i, controlSamps=%i, exonsNB=%i",
                    clustID, len(clusterSampsIndexes), len(sampsInd) - len(clusterSampsIndexes), len(sourceExons))

        ################
        ### Call process
        # fit an exponential distribution from all intergenic regions (CN0)
        try:
            (meanIntergenicFPM, loc, scale) = fitExponential(intergenicsFPM[:, sampsInd])
        except Exception as e:
            logger.error("fitExponential failed for cluster %i : %s", clustID, repr(e))
            raise

        if plotFolder:  # DEBUG tracking plot
            try:
                preprocessExponFitPlot(clustID, meanIntergenicFPM, loc, scale,
                                       os.path.join(plotFolders[0], "cluster" + str(clustID) + "_ExponentialFit_coverageProfile.pdf"))
            except Exception as e:
                logger.error("preprocessExponFitPlot failed for cluster %i : %s", clustID, repr(e))
                raise

        # Calculating the threshold in FPM equivalent to 99% of the CDF (PPF = percent point function)
        unCaptFPMLimit = scipy.stats.expon.ppf(fracCDFExp, loc=loc, scale=scale)

        # Browse cluster-specific exons
        for exInd in range(len(sourceExons)):
            params = []
            params.append({'distribution': scipy.stats.expon, 'loc': loc, 'scale': scale})
            exonDefinition = exons[sourceExons[exInd]]
            exonFPM = exonsFPM[sourceExons[exInd], sampsInd]

            # DEBUG tracking, print progress every 10000 exons
            if exInd % 10000 == 0:
                logger.info("cluster n°%i - exonNb %i, %s", clustID, exInd, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

            ################
            ### exons filters
            ### Filter n°1: not captured => median coverage of the exon = 0
            if filterUncapturedExons(exonFPM):
                if plotFolder:
                    try:
                        preprocessExonProfilePlot("notCaptured", exonsFiltersSummary, clustID, [loc, scale],
                                                unCaptFPMLimit, sourceExons[exInd], exonDefinition, exonFPM,
                                                plotFolders[1])
                    except Exception as e:
                        raise
                continue

            ### robustly fit a gaussian (CN2)
            ### Filter n°2: fitting is impossible (median close to 0)
            try:
                (mean, stdev) = fitRobustGaussian(exonFPM)
                params.append({'distribution': scipy.stats.norm, 'loc': mean, 'scale': stdev})
            except Exception as e:
                if str(e) == "cannot fit":
                    exonsFiltersSummary["cannotFitRG"] += 1
                    continue
                else:
                    raise Exception("fitRobustGaussian %s", repr(e))

            ### Filter n°3: fitted gaussian overlaps the threshold associated with the uncaptured exon profile
            if filterZscore(mean, stdev, unCaptFPMLimit):
                if plotFolder:
                    try:
                        preprocessExonProfilePlot("RGClose2LowThreshold", exonsFiltersSummary, clustID, [loc, scale],
                                                unCaptFPMLimit, sourceExons[exInd], exonDefinition, exonFPM,
                                                plotFolders[2], [mean, stdev])
                    except Exception as e:
                        raise
                continue

            ### Filter n°4: the samples contribution rate to the gaussian is too low (<50%)
            if filterSampsContrib2Gaussian(mean, stdev, exonFPM):
                if plotFolder:
                    try:
                        preprocessExonProfilePlot("fewSampsInRG", exonsFiltersSummary, clustID, [loc, scale],
                                                unCaptFPMLimit, sourceExons[exInd], exonDefinition, exonFPM,
                                                plotFolders[3], [mean, stdev])
                    except Exception as e:
                        raise
                continue

            exonsFiltersSummary["exonsCalls"] += 1

            params.append({'distribution': scipy.stats.norm, 'loc': mean / 2, 'scale': stdev})
            params.append({'distribution': scipy.stats.norm, 'loc': 3 * (mean / 2), 'scale': stdev})

            ################
            ### exon pass filters : compute likelihood for each sample
            # only cluster-samples  
            likelihoods = sampFPM2Probs(exonFPM, clusterSampsIndexes, sampsInd, params)
            clusterCalls[exInd] = likelihoods

            #     # graphic representation of callable exons for interest samples
            #     sampName = clusterSampsNames[i]
            #     if (not samps2Check) or (sampName not in samps2Check):
            #         continue      
            #     if plotFolder:
            #         try:
            #             preprocessExonProfilePlot("exonsCalls", exonsFiltersSummary, clustID, [loc, scale],
            #                                     unCaptFPMLimit, sourceExons[exInd], exonDefinition, exonFPM,
            #                                     plotFolders[4], [mean, stdev], [sampName, sampFPM, likelihoods])
            #         except Exception as e:
            #             raise

        if plotFolder:
            figures.plots.plotPieChart(clustID, exonsFiltersSummary, os.path.join(plotFolders[0], "cluster" + str(clustID) + "_filtersSummary_pieChart.pdf"))

        print(timer.timer_dict)

        return (clustID, colIndInCNCallsArray, sourceExons, clusterCalls)

    except Exception as e:
        logger.error("CNCalls failed for cluster n°%i - %s", clustID, repr(e))
        raise Exception(str(clustID))


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################



############################################
# makePlotDir
# Given the plot folder path, cluster ID, list of sample names associated with the cluster,
# and a user-defined list of sample names to check, the function creates the necessary
# directories for storing plots.
# Creates:
# - A cluster directory with the specified cluster ID in the provided plot folder.
# - Subdirectories within the cluster directory if there is an intersection between the
#   cluster sample names and the sample names to check. The subdirectories include
#   "F1_median", "F3_zscore", "F4_weight", and "PASS".
#
# Args:
# - plotDir [str]: path to the directory where the cluster directories should be created
# - clustID [int]
# - clusterSampsNames (list[str]): list of sample names associated with the cluster
# - samps2Check (list[str]): user-defined list of sample names to check
#
# Returns a list of paths to the created subdirectories.
@timer.measure_time
def makePlotFolders(plotFolder, clustID, clusterSampsNames, samps2Check):
    # To fill and returns
    pathPlotFolders = []

    ###############
    # Path to the cluster directory
    clustFolder = os.path.join(plotFolder, "cluster_" + str(clustID))
    try:
        os.makedirs(clustFolder, exist_ok=True)
    except OSError:
        raise Exception("Error creating directory " + clustFolder)

    pathPlotFolders.append(clustFolder)

    ##############
    # Path to filters directories
    plotPaths = ["F1_median", "F3_zscore", "F4_weight", "PASS"]
    for folderNameInd in range(len(plotPaths)):
        if folderNameInd == len(plotPaths):
            if not bool(set(clusterSampsNames) & set(samps2Check)):
                continue

        path = os.path.join(clustFolder, plotPaths[folderNameInd])
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            raise Exception("Error creating directory " + path)
        pathPlotFolders.append(path)

    return pathPlotFolders


#############################################################
# getCTRLSamps
#
# Args:
# - ctrlClusterIDs (list[int]): reference cluster IDs
# - clusters (list of lists[ints]): each list corresponds to a cluster index containing the
#                                   associated "samples" indexes.
#
# Return an int list of control "samples" indexes.
@timer.measure_time
def getCTRLSamps(ctrlClusterIDs, clusters):
    ctrlSamps = []
    for ctrlClusterID in ctrlClusterIDs:
        ctrlSamps.extend(clusters[ctrlClusterID].copy())
    return ctrlSamps


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
    loc, scale = scipy.stats.expon.fit(meanIntergenicFPM)

    return (meanIntergenicFPM, loc, scale)


###########################
# preprocessExponFitPlot
# Generates an exponential fit plot for a given cluster using the provided data and parameters.
# It performs data smoothing, calculates the probability density function for the exponential
# distribution, and plots the exponential fit.
#
# Args:
# - clustID [int]: The ID of the cluster.
# - meanIntergenicFPM (list[floats]): The list of mean intergenic FPM values.
# - loc (float), scale (float): parameters of the exponential distribution.
# - folder (str): The path to the folder where the plot should be saved.
#
# Return None
@timer.measure_time
def preprocessExponFitPlot(clustID, meanIntergenicFPM, loc, scale, file):
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
    makePDF(getattr(scipy.stats, 'expon'), [loc, scale], dr, yLists)
    plotLegs.append("expon={:.2f}, {:.2f}".format(loc, scale))

    figures.plots.plotExponentialFit(plotTitle, dr, yLists, plotLegs, file)


################################################
# makePDF
# Generate distributions's Probability Distribution Function (likelihood) and added it to yLists
# Args:
# - distribName (scipy object): probability distribution function from the SciPy library
# - params (list[floats]): parameters used to build the probability distribution function.
# - rangeData (list[floats]): range of FPM data
# - yLists (list[floats]): computed PDF values are appended
@timer.measure_time
def makePDF(param, rangeData, yLists):
    # Separate parts of parameters
    distribution = param['distribution']
    loc = param['loc']
    scale = param['scale']

    # Build PDF
    x = rangeData
    y = distribution.pdf(x, loc=loc, scale=scale)
    yLists.append(y)


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
# Given a sample FPM value for an exon and parameters of an exponential and a Gaussian distribution,
# calculation of likelihood (value of the PDF) for each type of copy number
#
# Args:
# - mean [float]
# - stdev [float]
# - loc, scale [float]: exponential distribution paramaters
# - exSampFPM [float]: FPM value for a sample and an exon
#
# Returns:
# - likelihoods (list[float]): likelihoods for each copy number (CN0,CN1,CN2,CN3+)
@timer.measure_time
def sampFPM2Probs(exonFPM, clusterSampsIndexes, sampsInd, params):
    clusterSampInd = [i for i, val in enumerate(sampsInd) if val in clusterSampsIndexes]
    exonFPM4Clust = exonFPM[clusterSampInd]
    # Liste pour stocker les PDF par groupe
    pdf_groups = []

    # Calcul des PDF pour chaque jeu de paramètres et chaque valeur de x
    for param in params:
        makePDF(param, exonFPM4Clust, pdf_groups)
    pdf_array = np.array(pdf_groups)
    concatenated_list = pdf_array.flatten(order='F')
    return concatenated_list

    # # CN2 mean shift to get CN1 mean
    # meanCN1 = mean / 2

    # # To Fill
    # # empty list to store the likelihoods for each copy number
    # likelihoods = np.zeros(4, dtype=np.float32)

    # # Calculate the likelihood for the exponential distribution (CN0)
    # # truncate at CN1 mean (exponential distribution  has a heavy tail)
    # cdfCN0Threshold = scipy.stats.expon.cdf(meanCN1, loc=loc, scale=scale)
    # if exSampFPM <= meanCN1:
    #     likelihoods[0] = (1 / cdfCN0Threshold) * scipy.stats.expon.pdf(exSampFPM, loc=loc, scale=scale)

    # # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    # likelihoods[1] = scipy.stats.norm.pdf(exSampFPM, meanCN1, stdev)
    # likelihoods[2] = scipy.stats.norm.pdf(exSampFPM, mean, stdev)
    # likelihoods[3] = scipy.stats.norm.pdf(exSampFPM, 3 * meanCN1, stdev)

    # return likelihoods


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
# given the information on the current cluster, the current exon and the current sample
# defines the variables for a graphical representation of the coverage profile with
# different laws fitting (exponential and Gaussian).
#
# Args:
# - status [str]: filter status
# - exonsFiltersSummary (dict): keys = filter status[str], value= exons counter
# - clustID [int]
# - exponParams (list[floats]): exponential distribution paramaters (loc, scale)
# - unCapturedThreshold [floats]: FPM threshold separating captured and non-captured exons
# - exInd [int]: "exons" index
# - exonInfos (list[str,int,int,str]): exon details [CHR,START,END,EXONID]
# - exonFPM (list[floats]): FPM counts from an exon
# - folder [str]: path to a pdf folder
# - gaussianParams (list[floats]): (optional) mean and stdev from robust fitting of Gaussian
# - sampInfos (composite lis): (optional) [sample name[str], FPM value[float], likelihoods(np.ndarray[floats])]
#
# Return None
@timer.measure_time
def preprocessExonProfilePlot(status, exonsFiltersSummary, clustID, exponParams, unCaptThreshold,
                              exInd, exonInfos, exonFPM, folder, gaussianParams=None, sampInfos=None):

    # Fixed parameter
    nbExLimit2Plot = 100  # No. of threshold multiplier exons = one plot

    if (status != "exonsCalls"):
        exonsFiltersSummary[status] += 1
        if exonsFiltersSummary[status] % nbExLimit2Plot != 0:
            return

    # initiate plots variables
    fileTitle = f"ClusterID_{clustID}_exonInd_{exInd}_{'_'.join(map(str, exonInfos))}"
    plotTitle = f"ClusterID n° {clustID} exon:{'_'.join(map(str, exonInfos))}"
    yLists = []
    plotLegs = ["expon={:.2f}, {:.2f}".format(exponParams[0], exponParams[1])]
    verticalLines = [unCaptThreshold]
    vertLinesLegs = ["UncoverThreshold={:.3f}".format(unCaptThreshold)]
    # creates FPM ranges base
    xi = np.linspace(0, max(exonFPM), 1000)
    # calculate the probability density function for the exponential distribution
    makePDF(getattr(scipy.stats, 'expon'), exponParams, xi, yLists)
    ylim = max(yLists[0]) / 10

    ##############
    # populate the plot variables according to the presence of the arguments passed to the function
    if gaussianParams is not None:
        # calculate the probability density function for the gaussian distribution (CN2)
        makePDF(getattr(scipy.stats, 'norm'), gaussianParams, xi, yLists)
        plotLegs.append("RG CN2={:.2f}, {:.2f}".format(gaussianParams[0], gaussianParams[1]))
        ylim = 2 * max(yLists[1])

        if sampInfos is not None:
            index_maxProb = max(enumerate(sampInfos[2]), key=lambda x: x[1])[0]
            if index_maxProb == 2:  # recurrent case
                return

            meanCN1 = gaussianParams[0] / 2
            # calculate the probability density function for the gaussian distribution (CN1)
            makePDF(getattr(scipy.stats, 'norm'), [meanCN1, gaussianParams[1]], xi, yLists)
            plotLegs.append("RG CN1={:.2f}, {:.2f}".format(meanCN1, gaussianParams[1]))

            # calculate the probability density function for the gaussian distribution (CN3)
            makePDF(getattr(scipy.stats, 'norm'), [3 * meanCN1, gaussianParams[1]], xi, yLists)
            plotLegs.append("RG CN3={:.2f}, {:.2f}".format(3 * meanCN1, gaussianParams[1]))

            verticalLines.append(sampInfos[1])
            vertLinesLegs.append(f"sample name={sampInfos[0]}")
            probs = ', '.join('{:.2e}'.format(x) for x in sampInfos[2])
            plotTitle += f"\nprobs={probs}"
            fileTitle = f"CN{index_maxProb}_" + sampInfos[0] + "_" + fileTitle

    PDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, fileTitle))
    figures.plots.plotExonProfile(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_File)
    PDF_File.close()
