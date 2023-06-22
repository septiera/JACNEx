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
# clustID [int]
# exonsFPM (np.ndarray[floats]): normalised counts from exons
# intergenicsFPM (np.ndarray[floats]): normalised counts from intergenic windows
# samples (list [str]): samples names
# exons (list of lists[str, int, int, str]): exon definitions [CHR,START,END,EXONID]
# clusters (list of lists[ints]): each list corresponds to a cluster index containing the
#                                 associated "samples" indexes
# ctrlsClusters (list of lists[int]): each list corresponds to a cluster index containing
#                                     control cluster index
# specClusters (list[int]): each list index = cluster index, values = autosomes (0)
#                           or gonosomes (1)
# CNTypes (list[str]): Copy number states.
# exonsBool (np.ndarray[bool]): indicates if each exon is in a gonosomal region (True) or not (False).
# outFile [str]: results calls file path
# sampsExons2Check (list of lists[int]): exon indices for each "sample" index to be graphically
#                                        validated.
#
# Returns a tuple (clustID, colInd4CNCallsArray, exonInd2Process, clusterCallsArray):
# - clustID [int]
# - colInd4CNCallsArray (list[ints]): column indexes for the CNcallsArray.
# - exonInd2Process (list[ints]): exons indexes (rows indexes) for the CNcallsArray.
# - clusterCallsArray (np.ndarray[floats]): The cluster calls array.


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# makeResFolders
# creates directories for storing plots associated with a cluster.
# It takes as input the cluster ID, the path to the main plot folder, a list of folder
# names to create, and a list of sample names associated with the cluster.
#
# Args:
# - clustID [int]: ID of the cluster.
# - resFolder [str]: path to the plot folder where the cluster directories should be created.
# - folderNames (list[str]): folder names to be created within each cluster directory only in DEBUG mode.
# - clusterSamps2Plot (list[str]): sample names targeted by the user and associated with the cluster
#
# Returns a tuple (clustFolder, pathPlotFolders):
# - clustFolder [str]: Path to the created cluster directory.
# - pathPlotFolders (list[str]): paths to the created subdirectories.
@timer.measure_time
def makeResFolders(clustID, resFolder, folderNames, clusterSamps2Plot):
    # To fill and returns
    pathPlotFolders = []

    clustFolder = os.path.join(resFolder, "cluster_" + str(clustID))
    try:
        os.makedirs(clustFolder, exist_ok=True)
    except OSError:
        raise Exception("Error creating directory " + clustFolder)

    if (logging.getLogger().getEffectiveLevel() <= logging.DEBUG):
        for folderNameInd in range(len(folderNames)):
            path = os.path.join(clustFolder, folderNames[folderNameInd])

            if folderNameInd == len(folderNames) - 1:
                # Check if the last folder should be created
                if not clusterSamps2Plot:
                    continue  # Skip creating the last folder if pathPlotFolders is empty

            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                raise Exception("Error creating directory " + path)

            pathPlotFolders.append(path)

    return (clustFolder, pathPlotFolders)


#############################################################
# getAllSampsIndex
# retrieve the sample indexes associated with a given cluster ID, including the
# sample indexes associated with its control cluster IDs.
#
# Args:
# - clustID [int]: The cluster ID for which to retrieve control sample indexes.
# - ctrlsClusters (list of lists[ints]): Each list corresponds to a cluster index and
#                                        contains the associated control sample indexes.
# - clusters (list of lists[ints]): Each list corresponds to a cluster index and contains
#                                   the associated sample indexes.
#
# Returns an integer list of control "samples" indexes
@timer.measure_time
def getAllSampsIndex(clustID, ctrlsClusters, clusters):
    ctrlClusterIndex = ctrlsClusters[clustID]
    sampsInd = clusters[clustID].copy()
    if (len(ctrlClusterIndex) != 0):
        ctrlSamps = []
        for ctrlIndex in ctrlClusterIndex:
            ctrlSamps.extend(clusters[ctrlIndex].copy())
        sampsInd = sampsInd + ctrlSamps

    return sampsInd


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
# - folder [str]: The path to the folder where the plot should be saved.
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
# Given exon FPM values, cluster sample indexes, sample indexes, and a list of parameter
# dictionaries as input.
#  - filters the exon FPM array to include only the samples associated with the cluster.
#  - initializes an empty list, `pdf_groups`, to store the PDF values for each parameter set.
#  - iterates over each parameter dictionary in the `params` list and calls the `makePDF`
#    function to calculate the PDF values for the filtered exon FPM data.
#  - appends PDF values to the `pdf_groups` list.
#  - `pdf_groups` list is converted to a numpy array, `pdf_array`, and returned.
#
# Args:
# - exonFPM (np.ndarray[floats]): Array of FPM values for one exon.
# - clusterSampsIndexes (list[int]): List of sample indexes associated with the cluster.
# - sampsInd (list[int]): sample indexes
# - params (list[dict]): parameter dictionaries for generating PDFs.
#
# Returns:
# - likelihoods (np.ndarray[floats]): PDF values for each set of parameters,
#                                     dim = (CN0,CN1,CN2,CN3+)*sampsNB
@timer.measure_time
def sampFPM2Probs(exonFPM, clusterSampsIndexes, sampsInd, params):
    clusterSampInd = [i for i, val in enumerate(sampsInd) if val in clusterSampsIndexes]
    exonFPM4Clust = exonFPM[clusterSampInd]
    # Liste pour stocker les PDF par groupe
    pdf_groups = []

    # Calcul des PDF pour chaque jeu de paramètres et chaque valeur de x
    for param in params:
        makePDF(param, exonFPM4Clust, pdf_groups)
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
# preprocesses and generates exon profile plots based on the provided parameters and data.
#   - checks the status of the exon filters and the summary to determine if the number of
# exons exceeds the limits for plotting. If the limits are exceeded, the function returns
# without generating the plots.
#   - retrieves the appropriate folder based on the provided status.
#   - initializes variables for file and plot titles, lists to store PDF values, plot legends,
# vertical lines, and their respective legends. It also creates a range of FPM values.
#   - populates the plot variables based on the presence of multiple parameters. If there
# are multiple parameters, PDFs are calculated for each parameter using the `makePDF` function.
# Plot legends and ylim values are updated accordingly.
#   - optionnal : clusterSamps2Plot is provided, iterates over the cluster sample indexes and performs
# additional calculations and plot updates.
#       - calculates PDFs for additional parameters, adds them to the plot legends,
# and adds sample-specific vertical lines and their legends. The function also updates
# the plot title and file title accordingly.
#   - generates the exon profile plot using the provided data and plot variables.
# The plot is saved in a PDF file, given the information on the current cluster, the current exon
# and the current sample, defines the variables for a graphical representation of the coverage profile with
# different laws fitting (exponential and Gaussian).
#
# Args:
# - status [str]: Status of the exon filters.
# - exonsFiltersSummary (dict): Summary of exon filters. keys = filter status[str], value= exons counter
# - clustID [int]: Cluster ID.
# - params (list[dict]): List of parameter dictionaries for generating PDFs.
# - unCaptThreshold [float]: FPM threshold separating captured and non-captured exons
# - exonInfos (list[int]): Exon information [CHR,START,END,EXONID].
# - exonFPM (np.ndarray[floats]): Array of FPM values for one exon.
# - folders (list[str]): folders for storing the generated plots.
# - samples (list[str], optional): sample names. Defaults to None.
# - clusterSamps2Plot (list[int], optional): sample indexes to plot. Defaults to None.
# - sampsInd (list[int], optional): all sample indexes. Defaults to None.
# - likelihoods (np.ndarray[floats], optional): likelihood values. Defaults to None.
#
# Return None
@timer.measure_time
def preprocessExonProfilePlot(status, exonsFiltersSummary, clustID, params,
                              unCaptThreshold, exonInfos, exonFPM, folders,
                              samples=None, clusterSamps2Plot=None, sampsInd=None, likelihoods=None):

    if not hasattr(preprocessExonProfilePlot, 'dictionnaire'):
        preprocessExonProfilePlot.dictionnaire = {}  # dictionnary init

    # Fixed parameter
    nbExLimit2PlotF2F4 = 10  # No. of threshold multiplier exons = one plot
    nbExLimit2PlotF1F3 = 10000

    if status in ["cannotFitRG", "fewSampsInRG"]:
        if exonsFiltersSummary[status] % nbExLimit2PlotF2F4 != 0:
            return
    elif status in ["notCaptured", "RGClose2LowThreshold"]:
        if exonsFiltersSummary[status] % nbExLimit2PlotF1F3 != 0:
            return

    for f in folders:
        if status in f:
            folder = f

    # initiate plots variables
    fileTitle = f"ClusterID_{clustID}_{'_'.join(map(str, exonInfos))}"
    plotTitle = f"ClusterID n° {clustID} exon:{'_'.join(map(str, exonInfos))}"
    yLists = []
    plotLegs = ["expon={:.2f}, {:.2f}".format(params[0]['loc'], params[0]['scale'])]
    verticalLines = [unCaptThreshold]
    vertLinesLegs = ["UncoverThreshold={:.3f}".format(unCaptThreshold)]
    # creates FPM ranges base
    xi = np.linspace(0, max(exonFPM), 1000)
    # calculate the probability density function for the exponential distribution
    makePDF(params[0], xi, yLists)
    ylim = max(yLists[0]) / 10

    ##############
    # populate the plot variables according to the presence of the arguments passed to the function
    if len(params) > 1:
        # calculate the probability density function for the gaussian distribution (CN2)
        makePDF(params[1], xi, yLists)
        plotLegs.append("RG CN2={:.2f}, {:.2f}".format(params[1]['loc'], params[1]['scale']))
        ylim = 2 * max(yLists[1])

        if clusterSamps2Plot is not None:
            for samp in clusterSamps2Plot:
                sampName = samples[samp]
                sampFPM = exonFPM[sampsInd.index(samp)]
                samplikelihood = likelihoods[:, sampsInd.index(samp)]
                index_maxProb = np.argmax(samplikelihood)

                if index_maxProb != 0:
                    if index_maxProb in preprocessExonProfilePlot.dictionnaire:
                        preprocessExonProfilePlot.dictionnaire[index_maxProb] += 1
                    else:
                        preprocessExonProfilePlot.dictionnaire[index_maxProb] = 1

                    if preprocessExonProfilePlot.dictionnaire[index_maxProb] % nbExLimit2PlotF1F3 != 0:
                        return

                # calculate the probability density function for the gaussian distribution (CN1)
                makePDF(params[2], xi, yLists)
                plotLegs.append("RG CN1={:.2f}, {:.2f}".format(params[2]['loc'], params[2]['scale']))

                # calculate the probability density function for the gaussian distribution (CN3)
                makePDF(params[3], xi, yLists)
                plotLegs.append("RG CN3={:.2f}, {:.2f}".format(params[3]['loc'], params[3]['scale']))

                verticalLines.append(sampFPM)
                vertLinesLegs.append(f"sample name={sampName}")
                probs = ', '.join('{:.2e}'.format(x) for x in samplikelihood)
                plotTitle += f"\nprobs={probs}"
                fileTitle = f"CN{index_maxProb}_" + sampName + "_" + fileTitle

    PDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, fileTitle))
    figures.plots.plotExonProfile(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_File)
    PDF_File.close()
