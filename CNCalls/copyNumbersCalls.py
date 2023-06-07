import logging
import os
import numba
import numpy as np
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time

import figures.plots

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# makePlotDir
# Create directories for storing plots related to a given cluster.
# Args:
# - plotDir [str]: path to the directory where the cluster directories should be created
# - clustID [int]
# Returns a list of paths to the created subdirectories
def makePlotDir(plotDir, clustID):
    # To fill and returns
    # List of paths to the created subdirectories
    pathDirPlotCN = []

    # Path to the cluster directory
    CNcallsClustDir = os.path.join(plotDir, "cluster_" + str(clustID))

    try:
        # Create the cluster directory if it doesn't exist
        os.makedirs(CNcallsClustDir, exist_ok=True)
    except OSError:
        # If there was an error creating the directory, raise an exception
        raise Exception("Error creating directory " + CNcallsClustDir)

    # List of subdirectories to create within the cluster directory
    PlotPaths = ["F1_median", "F3_zscore", "F4_weight", "PASS", "filteringRes_PieChart"]

    # Create each subdirectory
    for plotPath in PlotPaths:
        path = os.path.join(CNcallsClustDir, plotPath)
        # Check if the subdirectory already exists
        if os.path.isdir(path):
            raise Exception("Directory " + path + " already exists")
        os.makedirs(path)
        pathDirPlotCN.append(path)

    # Return the list of paths to the created subdirectories
    return pathDirPlotCN


#####################################
# CNCalls
# Given a cluster identifier, two arrays containing FPM values for intergenic regions and exons,
# as well as lists of definitions for samples and clusters, this function calculates the
# probabilities (likelihood) for each copy number type (CN0, CN1, CN2, CN3+) for each exon
# and each sample.
#
# Specifically:
# -Retrieves cluster-specific data such as exon and intergenic FPM (fragments per million)
# count data, and creates a list of exon information to process.
# -Fits an exponential distribution from all intergenic FPM data to obtain the law parameters
# and a FPM threshold corresponding to 99% of the cumulative distribution function (CDF).
# -Filters out exons that are not interpretable and fits a robust Gaussian:
#   -F1: filters exons with median coverage of 0, i.e., not captured.
#   -F2: filters exons where it is impossible to fit a robust Gaussian, i.e., median coverage
#        is close to 0.
#   -F3: filters exons where the fitted Gaussian overlaps the threshold associated with the
#        uncaptured exon profile.
#   -F4: filters exons where the sample contribution rate to the robust Gaussian is too low,
#        i.e., less than 50%.
# -Calculates probabilities for each sample/exon to determine CN0 (case where the sample FPM
# value is lower than the CN1 mean, otherwise 0), CN1, CN2, and CN3 and fills CNcallsArray
# -Produces filtered and unfiltered exon profiles and pie charts representations if a path
# for plotted to was defined by the user, saved in different PDF files.
#
# Args:
# - CNcallsArray (np.ndarray): exons copy number calls for each sample and exon, unfilled value = -1
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
# Returns CNcallsArray updated
def CNCalls(CNCallsArray, clustID, exonsFPM, intergenicsFPM,
            samples, exons, clusters, ctrlsClusters,
            sourceClusters, maskSourceExons, plotFolders, samps2Check):

    clusterSampsInd = clusters[clustID].copy()

    # If there are control clusters, add their samples indexes in a new list, else new list = previous list
    if len(ctrlsClusters[clustID]) != 0:
        ctrlClusterSampsInd = getCTRLSamps(ctrlsClusters[clustID], clusters)
        samps = clusterSampsInd + ctrlClusterSampsInd
    else:
        samps = clusterSampsInd

    exInd2Process = np.where(maskSourceExons == sourceClusters[clustID])[0]  # obtain the indexes of the cluster's source exons

    logger.info("clustSamps= %i, controlSamps= %i, exonsNB= %i",
                len(clusterSampsInd), len(samps) - len(clusterSampsInd), len(exInd2Process))

    # counter dictionary for each filter, only used to plot the pie chart
    exonsFiltersSummary = dict.fromkeys(["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "exonsCalls"], 0)

    # fit an exponential distribution from all pseudo exons
    try:
        (loc, scale) = fitExponential(intergenicsFPM[:, samps])
        logger.info("exponential params loc= %.2e scale=%.2e", loc, scale)
    except Exception as e:
        logger.error("fitExponential failed for cluster %i : %s", clustID, repr(e))
        raise Exception("fitExponential failed")

    # Calculating the threshold in FPM equivalent to 99% of the CDF (PPF = percent point function)
    unCaptFPMLimit = scipy.stats.expon.ppf(0.99, loc=loc, scale=scale)

    # Browse cluster-specific exons
    for exInd in range(len(exInd2Process)):
        exIndInFullTab = exInd2Process[exInd]
        exonInfos = exons[exIndInFullTab]
        exonFPM = exonsFPM[exIndInFullTab]

        # Print progress every 10000 exons
        if exInd % 10000 == 0:
            logger.info(" - exonNb %i, %s", exInd, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # Filter n°1: not captured => median coverage of the exon = 0
        if filterUncapturedExons(exonFPM):
            preprocessPlotData("notCaptured", exonsFiltersSummary, clustID, loc, scale, unCaptFPMLimit,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[0] if plotFolders else None)
            continue

        # robustly fit a gaussian
        # Filter n°2: fitting is impossible (median close to 0)
        try:
            (mean, stdev) = fitRobustGaussian(exonFPM)
        except Exception as e:
            if str(e) == "cannot fit":
                continue
            else:
                raise e

        # Filter n°3: RG overlaps the threshold associated with the uncaptured exon profile
        if filterZscore(mean, stdev, unCaptFPMLimit):
            preprocessPlotData("RGClose2LowThreshold", exonsFiltersSummary, clustID, loc, scale, unCaptFPMLimit,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[1] if plotFolders else None, mean, stdev)  # !!!! to change
            continue

        # Filter n°4: the sample contribution rate to the robust Gaussian is too low (<50%)
        if filterSampsContrib2Gaussian(mean, stdev, exonFPM):
            preprocessPlotData("fewSampsInRG", exonsFiltersSummary, clustID, loc, scale, unCaptFPMLimit,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[2] if plotFolders else None, mean, stdev)  # !!!! to change
            continue

        exonsFiltersSummary["exonsCalls"] += 1

        # Browse cluster-samples
        for i in range(len(samps)):
            sampInd = samps[i]
            if sampInd not in clusterSampsInd:  # already processed
                continue

            # density probabilities for each copy number
            sampFPM = exonFPM[i]
            probs = sampFPM2Probs(mean, stdev, loc, scale, unCaptFPMLimit, sampFPM)

            fillCNCallsArray(CNCallsArray, samps[i], exIndInFullTab, probs)

            # graphic representation of callable exons for interest patients
            sampName = samples[sampInd]
            if (not samps2Check) or (sampName not in samps2Check):
                continue
            preprocessPlotData("exonsCalls", exonsFiltersSummary, clustID, loc, scale, unCaptFPMLimit,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[3] if plotFolders else None, mean, stdev,
                               [sampName, sampFPM, probs])  # !!!! to change

    if plotFolders:
        pieFile = matplotlib.backends.backend_pdf.PdfPages(os.path.join(plotFolders[4], "pieChart_Filtering_cluster" + str(clustID) + ".pdf"))
        figures.plots.plotPieChart(clustID, exonsFiltersSummary, pieFile)
        pieFile.close()

    return CNCallsArray


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################################################
# getCTRLSamps
#
# Args:
# - ctrlClusterIDs (list[int]): reference cluster IDs
# - clusters (list of lists[ints]): each list corresponds to a cluster index containing the
#                                   associated "samples" indexes.
#
# Return an int list of control "samples" indexes.
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
# Returns a tuple (loc, scale), parameters of the exponential
def fitExponential(intergenicsFPMClust):
    # compute meanFPM for each intergenic region (speed up)
    meanIntergenicFPM = np.mean(intergenicsFPMClust, axis=1)

    # Fit an exponential distribution, imposing location = 0
    # f(x, scale) = (1/scale)*exp(-x/scale)
    loc, scale = scipy.stats.expon.fit(meanIntergenicFPM, floc=0)

    return (loc, scale)


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
# Given a robustly fitted gaussian parameters and an FPM threshold separating
# uncaptured exons from captured exons, exon are filtered when the capture profile
# is indistinguishable from the non-capture profile.
#
# Spec:
# - setting a tolerated deviation threshold, bdwthThreshold
# - check that the standard deviation is not == 0 otherwise no pseudo zscore
# can be calculated, change it if necessary
# - pseudo zscore calculation
# - comparison pseudo zscore with the tolerated deviation threshold => filtering
#
# Args:
# - mean [float], stdev [float]: parameters of the normal
# - unCaptFPMLimit [float]: FPM threshold separating captured and non-captured exons
#
# Returns "True" if exon doesn't pass the filter otherwise "False"
@numba.njit
def filterZscore(mean, stdev, unCaptFPMLimit):
    # Fixed paramater
    bdwthThreshold = 3  # tolerated deviation threshold

    # mean != 0 and all samples have the same coverage value.
    if (stdev == 0):
        stdev = mean / 20  # simulates 5% on each side of the mean

    # meanRG != 0 because of filter n°1 => stdevRG != 0
    zscore = (mean - unCaptFPMLimit) / stdev

    if (zscore < bdwthThreshold):
        return True
    else:
        return False


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
# - unCapturedThreshold [float]: FPM threshold separating captured and non-captured exons
# - exSampFPM [float]: FPM value for a sample and an exon
#
# Returns:
# - likelihoods (list[float]): likelihoods for each copy number (CN0,CN1,CN2,CN3+)
def sampFPM2Probs(mean, stdev, loc, scale, unCaptThreshold, exSampFPM):
    # CN2 mean shift to get CN1 mean
    meanCN1 = mean / 2

    # To Fill
    # empty list to store the likelihoods for each copy number
    likelihoods = np.zeros(4, dtype=np.float32)

    # Calculate the likelihood for the exponential distribution (CN0)
    # truncate at CN1 mean (exponential distribution  has a heavy tail)
    cdf_cno_threshold = scipy.stats.expon.cdf(unCaptThreshold, loc=loc, scale=scale)
    if exSampFPM <= meanCN1:
        likelihoods[0] = (1 / cdf_cno_threshold) * scipy.stats.expon.pdf(exSampFPM, loc=loc, scale=scale)

    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    likelihoods[1] = scipy.stats.norm.pdf(exSampFPM, meanCN1, stdev)
    likelihoods[2] = scipy.stats.norm.pdf(exSampFPM, mean, stdev)
    likelihoods[3] = scipy.stats.norm.pdf(exSampFPM, 3 * meanCN1, stdev)

    return likelihoods


#############################################################
# fillCNCallsArray
# Arguments:
# - CNCallsArray (np.ndarray[floats]): copy number call likelihood
# - sampIndex [int]: "samples" index
# - exIndex [int]: exon index in CNCallsArray
# - probs (list[floats]): copy number call likelihood [CN0,CN1,CN2,CN3]
def fillCNCallsArray(CNCallsArray, sampIndex, exIndex, probs):
    sampArrayInd = sampIndex * 4
    # sanity check : don't overwrite data
    if np.sum(CNCallsArray[exIndex, sampArrayInd:sampArrayInd + 4]) < 0:
        # Populate the CNcallsArray with the copy number call likelihood probabilities
        CNCallsArray[exIndex, sampArrayInd:sampArrayInd + 4] = probs
    else:
        raise Exception("overwrite calls in CNCallsArray")


#########################
# preprocessPlotData
# given the information on the current cluster, the current exon and the current sample
# defines the variables for a graphical representation of the coverage profile with
# different laws fitting (exponential and Gaussian).
# Args:
# - status [str]: filter status
# - exonsFiltersSummary (dict): keys = filter status[str], value= exons counter
# - clustID [int]
# - exponParams (list[floats]): exponential distribution paramaters (loc, scale)
# - unCapturedThreshold [floats]: FPM threshold separating captured and non-captured exons
# - exInd [int]: "exons" index
# - exonInfos (list[str,int,int,str]): exon details [CHR,START,END,EXONID]
# - exonFPM (list[floats]): FPM counts from an exon
# - folder [str]: path to a folder where to save the pdfs
# - RGParams (list[floats]): (optional) mean and stdev from robust fitting of Gaussian
# - sampInfos (composite lis): (optional) [sample name[str], FPM value[float], likelihoods(np.ndarray[floats])]
def preprocessPlotData(status, exonsFiltersSummary, clustID, exponParams, unCaptThreshold,
                       exInd, exonInfos, exonFPM, folder, RGParams=None, sampInfos=None):

    # Fixed parameter
    nbExLimit2Plot = 100  # No. of threshold multiplier exons = one plot

    if (status != "exonsCalls"):
        exonsFiltersSummary[status] += 1
        if exonsFiltersSummary[status] % nbExLimit2Plot != 0:
            return

    if folder is None:
        return

    # initiate plots variables
    figTitle = f"ClusterID_{clustID}_exonInd_{exInd}_{'_'.join(map(str, exonInfos))}"
    plotTitle = f"ClusterID n° {clustID} exon:{'_'.join(map(str, exonInfos))}"
    yLists = []
    plotLegs = ["\nexpon={:.2f}, {:.2f}".format(exponParams[0], exponParams[1])]
    verticalLines = [unCaptThreshold]
    vertLinesLegs = ["UncoverThreshold={:.3f}".format(unCaptThreshold)]
    # creates FPM ranges base
    xi = np.linspace(0, max(exonFPM), 1000)
    # calculate the probability density function for the exponential distribution
    makePDF(getattr(scipy.stats, 'expon'), exponParams, xi, yLists)
    ylim = max(yLists[0]) / 10

    ##############
    # populate the plot variables according to the presence of the arguments passed to the function
    if RGParams is not None:
        # calculate the probability density function for the gaussian distribution (CN2)
        makePDF(getattr(scipy.stats, 'norm'), RGParams, xi, yLists)
        plotLegs.append(f"\nRG CN2={RGParams[0]:.2f}, {RGParams[1]:.2f}")
        ylim = 2 * max(yLists[1])

        if sampInfos is not None:
            index_maxProb = max(enumerate(sampInfos[2]), key=lambda x: x[1])[0]
            if index_maxProb == 2:  # recurrent case
                return

            meanCN1 = RGParams[0] / 2
            # calculate the probability density function for the gaussian distribution (CN1)
            makePDF(getattr(scipy.stats, 'norm'), [meanCN1, RGParams[1]], xi, yLists)
            plotLegs.append(f"\nRG CN1={meanCN1:.2f}, {RGParams[1]:.2f}")

            # calculate the probability density function for the gaussian distribution (CN3)
            makePDF(getattr(scipy.stats, 'norm'), [3 * meanCN1, RGParams[1]], xi, yLists)
            plotLegs.append(f"\nRG CN3={3 * meanCN1:.2f}, {RGParams[1]:.2f}")

            verticalLines.append(sampInfos[1])
            vertLinesLegs.append(f"sample name={sampInfos[0]}")
            probs = ', '.join([f"{inner:.2e}" for inner in sampInfos[2].astype(str)])
            plotTitle += f"\nprobs={probs}"
            figTitle = f"CN{index_maxProb}_" + sampInfos[0] + "_" + figTitle

    PDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, figTitle))
    figures.plots.plotExonProfil(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_File)
    PDF_File.close()


################################################
# makePDF
# Generate distributions's Probability Distribution Function and added it to yLists
# Args:
# - distribName (scipy object): probability distribution function from the SciPy library
# - params (list[floats]): parameters used to build the probability distribution function.
# - rangeData (list[floats]): range of FPM data
# - yLists (list[floats]): computed PDF values are appended
def makePDF(distribName, params, rangeData, yLists):
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Build PDF
    x = rangeData
    y = distribName.pdf(x, loc=loc, scale=scale, *arg)
    yLists.append(y)
