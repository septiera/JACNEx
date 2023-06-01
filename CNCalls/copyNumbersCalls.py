import logging
import os
import numba
import numpy as np
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time

import clusterSamps.smoothing
import clusterSamps.getGonosomesExonsIndexes
import clusterSamps.qualityControl
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
# Given a cluster identifier, two arrays containing FPM values for pseudo-exons and exons,
# as well as lists of definitions for samples and clusters, this function calculates the
# probabilities (likelihood) for each copy number type (CN0, CN1, CN2, CN3+) for each exon
# and each sample.
# Specifically:
# -Retrieves cluster-specific data such as exon and intergenic FPM (fragments per million)
# count data, and creates a list of exon information to process.
# -Fits an exponential distribution from all intergenic FPM data to obtain the law parameters
# and a FPM threshold corresponding to 99% of the cumulative distribution function (CDF).
# -Filters out exons that are not interpretable and fits a robust Gaussian:
#   -F1: filters exons with median coverage of 0, i.e., not captured.
#   -F2: filters exons where it is impossible to fit a robust Gaussian, i.e., median coverage is close to 0.
#   -F3: filters exons where the fitted Gaussian overlaps the threshold associated with the uncaptured exon profile.
#   -F4: filters exons where the sample contribution rate to the robust Gaussian is too low, i.e., less than 50%.
# -Calculates probabilities for each sample/exon to determine CN0 (case where the sample FPM value is lower
# than the CN1 mean, otherwise 0), CN1, CN2, and CN3 and fills CNcallsArray
# -Produces filtered and unfiltered exon profiles and pie charts representations if a path for plotted
# to was defined by the user, saved in different PDF files.
# Args:
# - CNcallsArray (np.ndarray): exons copy number calls for each sample and exon, unfilled value = -1
# - clustID [int]
# - exonsFPM (np.ndarray[float]): normalised counts from exons
# - intergenicsFPM (np.ndarray[float]): normalised counts from intergenic windows
# - samples (list [str]): samples names
# - exons (list of lists[str, int, int, str]): exon definitions [CHR,START,END,EXONID]
# - sampsInClusts (list of lists[int]): "samples" indexes for each cluster
# - ctrlsInClusts (list of lists[int]): IDs of control clusters for each cluster
# - specClusts (list[int]): specifies if each cluster is from autosomes (0) or gonosomes (1)
# - maskGExIndexes (np.ndarray[bool]): indicates if each exon is in a gonosomal region (True) or not (False)
# - plotFolder (str): subdir (created if needed) where result plots files will be produced
# - sampsToCheck (str): sample names for the graphic control of the number of copies called
# Returns CNcallsArray updated
def CNCalls(CNCallsArray, clustID, exonsFPM, intergenicsFPM, samples, exons, sampsInClusts, ctrlsInClusts, specClusts, maskGExIndexes, plotFolders, samps2Check):
    clustSpeSamps = sampsInClusts[clustID].copy()

    # If there are control clusters, add their samples indexes in a new list, else new list = previous list
    if len(ctrlsInClusts[clustID]) != 0:
        refSamps = getSampsFromRefClusters(ctrlsInClusts[clustID], sampsInClusts)
        samps2Compare = clustSpeSamps + refSamps
    else:
        samps2Compare = clustSpeSamps

    exInd2Process = getExInd2Process(specClusts[clustID], maskGExIndexes)
    logger.info("clustSamps= %i, controlSamps= %i, exonsNB= %i",
                len(clustSpeSamps), len(samps2Compare) - len(clustSpeSamps), len(exInd2Process))

    # counter dictionary for each filter, only used to plot the pie chart
    exonsFiltersSummary = dict.fromkeys(["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "exonsCalls"], 0)

    # fit an exponential distribution from all pseudo exons
    try:
        (exponParams, unCaptThreshold) = fitExponential(intergenicsFPM[:, samps2Compare])
        logger.info("exponential params loc= %.2e scale=%.2e, uncaptured threshold=%.2e FPM",
                    exponParams[0], exponParams[1], unCaptThreshold)
    except Exception as e:
        logger.error("fitExponential failed for cluster %i : %s", clustID, repr(e))
        raise Exception("fitExponential failed")

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
            preprocessPlotData("notCaptured", exonsFiltersSummary, clustID, exponParams, unCaptThreshold,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[0] if plotFolders else None)
            continue

        # robustly fit a gaussian
        # Filter n°2: fitting is impossible (median close to 0)
        try:
            RGParams = fitRobustGaussian(exonFPM)
        except Exception as e:
            if str(e) == "cannot fit":
                continue
            else:
                raise e

        # Filter n°3: RG overlaps the threshold associated with the uncaptured exon profile
        if filterZscore(RGParams[0], RGParams[1], unCaptThreshold):
            preprocessPlotData("RGClose2LowThreshold", exonsFiltersSummary, clustID, exponParams, unCaptThreshold,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[1] if plotFolders else None, RGParams)
            continue

        # Filter n°4: the sample contribution rate to the robust Gaussian is too low (<50%)
        if filterSampsContribRG(exonFPM, RGParams[0], RGParams[1]):
            preprocessPlotData("fewSampsInRG", exonsFiltersSummary, clustID, exponParams, unCaptThreshold,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[2] if plotFolders else None, RGParams)
            continue

        exonsFiltersSummary["exonsCalls"] += 1

        # Browse cluster-samples
        for i in range(len(samps2Compare)):
            sampInd = samps2Compare[i]
            if sampInd not in clustSpeSamps:  # already processed
                continue

            # density probabilities for each copy number
            sampFPM = exonFPM[i]
            probs = sampFPM2Probs(RGParams[0], RGParams[1], exponParams, unCaptThreshold, sampFPM)

            fillCNCallsArray(CNCallsArray, samps2Compare[i], exIndInFullTab, probs)

            # graphic representation of callable exons for interest patients
            sampName = samples[sampInd]
            if (not samps2Check) or (sampName not in samps2Check):
                continue
            preprocessPlotData("exonsCalls", exonsFiltersSummary, clustID, exponParams, unCaptThreshold,
                               exIndInFullTab, exonInfos, exonFPM, plotFolders[3] if plotFolders else None, RGParams,
                               [sampName, sampFPM, probs])

    if plotFolders:
        pieFile = matplotlib.backends.backend_pdf.PdfPages(os.path.join(plotFolders[4], "pieChart_Filtering_cluster" + str(clustID) + ".pdf"))
        figures.plots.plotPieChart(clustID, exonsFiltersSummary, pieFile)
        pieFile.close()

    return CNCallsArray


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################################################
# getSampsFromRefClusters
# Args:
# - refClusterIDs (list[int]): reference cluster IDs
# - sampsInClusts (list of list[int]): each sub-list is composed of the samples for a given
# cluster ID (list index)
# Return:
# - refSamps: the ref sample list from reference cluster(s)
def getSampsFromRefClusters(refClusterIDs, sampsInClusts):
    refSamps = []
    for refID in refClusterIDs:
        refSamps.extend(sampsInClusts[refID].copy())
    return refSamps


#############################################################
# getExInd2Process
# Args:
# - clustSexStatus [bool]: cluster analyzed with 0 = autosomal exons, 1 = gonosomal exons
# - maskGExIndexes [np.ndarray[bool]]: for each "exon" index sexual status indicated (0: autosomes, 1: gonosomes)
# Return:
# - exInd2Process (list[int]): "exons" indexes
def getExInd2Process(clustSexStatus, maskGExIndexes):
    exInd2Process = []
    if clustSexStatus == 0:
        exInd2Process.extend(np.where(~maskGExIndexes)[0].tolist())
    else:
        exInd2Process.extend(np.nonzero(maskGExIndexes)[0].tolist())
    return exInd2Process


#############################################################
# fitExponential
# Given a counts array for a specific cluster identifier, fits an exponential distribution
# and deduces an FPM threshold separating the FPM values associated with non-capture from those captured.
# Spec:
# - coverage profile smoothing deduced from FPM averages per pseudo-exons = non-captured profile
# - exponential fitting
# - FPM threshold calculation from 99% of the cumulative distribution function
# Args:
# - intergenicsFPMClust (np.ndarray[floats]): FPM array specific to a cluster
# Returns a tupple (expParams, UncoverThreshold), each variable is created here:
# - expParams [tuple of floats]: estimated parameters (x2) of the exponential distribution
# - UncoverThreshold [float]: threshold for separating covered and uncovered regions,
# which corresponds to the FPM value at 99% of the CDF
def fitExponential(intergenicsFPMClust):
    # Fixed parameter seems suitable for smoothing
    bandWidth = 0.5
    #### To Fill and returns:
    expParams = []
    # vertical dashed lines: coordinates (default to 0 ie no lines) and legends
    UncoverThreshold = 0
    # compute meanFPM by intergenic regions
    # save computation time instead of taking the raw data (especially for clusters
    # with many samples)
    meanIntergenicFPM = np.mean(intergenicsFPMClust, axis=1)
    # Smoothing the average coverage profile of intergenic regions allows fitting the exponential
    # to sharper densities than the raw data
    try:
        (dr, dens, bwValue) = clusterSamps.smoothing.smoothData(meanIntergenicFPM, maxData=max(meanIntergenicFPM), bandwidth=bandWidth)
    except Exception as e:
        logger.error('smoothing failed for %s : %s', str(bandWidth), repr(e))
        raise
    # Fitting the exponential distribution and retrieving associated parameters => tupple(loc, scale)
    # f(x, scale) = (1/scale)*exp(-x/scale)
    # scale parameter is the inverse of the rate parameter (lambda) used in the mathematical definition
    # of the distribution.
    # Location parameter is an optional parameter that controls the location of the distribution
    # on the x-axis (fixed to 0 = floc).
    expParams = scipy.stats.expon.fit(dens, floc=0)
    # Calculating the threshold in FPM equivalent to 99% of the CDF (PPF = percent point function)
    UncoverThreshold = scipy.stats.expon.ppf(0.99, *expParams)

    return (expParams, UncoverThreshold)


###################
# FilterUncapturedExons
# Filter n°1: exon not covered in most samples.
# Given a FPM counts from an exon, calculating the median coverage and filter several possible cases:
# - all samples in the cluster haven't capture for the current exon
# - more than 2/3 of the samples have no capture.
# Warning: Potential presence of homodeletions. We have chosen don't call them because they affect too many samples
# Args:
# - exonFPM (list[floats]): FPM counts from an exon
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
# Args:
# - X (np.array): A sample of 1-dimensional mixture of gaussian random variables
# - mu (float, optional): Expectation. Defaults to None.
# - sigma (float, optional): Standard deviation. Defaults to None.
# - bandwidth (float, optional): Hyperparameter of truncation. Defaults to 2.
# - eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.
# Returns:
# - mu [float],sigma [float]: mean and stdev of the gaussian component
def fitRobustGaussian(X, mu=None, sigma=None, bandwidth=2.0, eps=1.0e-5):
    if mu is None:
        # median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)
    mu_0 = mu + 1

    if sigma is None:
        # rule of thumb
        sigma = np.std(X) / 3
    sigma_0 = sigma + 1

    bandwidth_truncated_normal_sigma = truncated_integral_and_sigma(bandwidth)

    while abs(mu - mu_0) + abs(sigma - sigma_0) > eps:
        # loop until tolerence is reached
        """
        create a uniform window on X around mu of width 2*bandwidth*sigma
        find the mean of that window to shift the window to most expected local value
        measure the standard deviation of the window and divide by the standard deviation of a truncated gaussian distribution
        measure the proportion of points inside the window, divide by the weight of a truncated gaussian distribution
        """
        Window = np.logical_and(X - mu - bandwidth * sigma < 0, X - mu + bandwidth * sigma > 0)

        # condition to identify exons with points arround at the median
        if Window.any():
            mu_0, mu = mu, np.average(X[Window])
            var = np.average(np.square(X[Window])) - mu**2
            sigma_0, sigma = sigma, np.sqrt(var) / bandwidth_truncated_normal_sigma
        # no points arround the median
        # e.g. exon where more than 1/2 of the samples have an FPM = 0.
        # A Gaussian fit is impossible => raise exception
        else:
            raise Exception("cannot fit")
    return ([mu, sigma])


#############################################################
# normal_erf
# ancillary function of the robustGaussianFitPrivate function computes Gauss error function
# The error function (erf) is used to describe the Gaussian distribution.
# It gives the probability that a random variable follows a given Gaussian distribution,
# indicating the probability that it is less than or equal to a given value.
# In other words, the error function quantifies the probability distribution for a
# random variable following a Gaussian distribution.
# this function replaces the use of the scipy.stats.erf module
def normal_erf(x, mu=0, sigma=1, depth=50):
    ele = 1.0
    normal = 1.0
    x = (x - mu) / sigma
    erf = x
    for i in range(1, depth):
        ele = - ele * x * x / 2.0 / i
        normal = normal + ele
        erf = erf + ele * x / (2.0 * i + 1)

    return np.clip(normal / np.sqrt(2.0 * np.pi) / sigma, 0, None), np.clip(erf / np.sqrt(2.0 * np.pi) / sigma, -0.5, 0.5)


#############################################################
# truncated_integral_and_sigma
# ancillary function of the robustGaussianFitPrivate function
# allows for a more precise and focused analysis of a function
# by limiting the study to particular parts of its defining set.
def truncated_integral_and_sigma(x):
    n, e = normal_erf(x)
    return np.sqrt(1 - n * x / e)


###################
# filterZscore
# Filter n°3: Gaussian(for CN2) is too close to unCaptThreshold.
# Given a robustly fitted gaussian paramaters and an FPM threshold separating
# uncaptured exons from captured exons, exon filtering when the capture profile
# is indistinguishable from the non-capture profile.
# Spec:
# - setting a tolerated deviation threshold, bdwthThreshold
# - check that the standard deviation is not == 0 otherwise no pseudo zscore
# can be calculated, change it if necessary
# - pseudo zscore calculation
# - comparison pseudo zscore with the tolerated deviation threshold => filtering
# Args:
# - mean [floats]
# - stdev [floats]
# - unCapturedThreshold [floats]: FPM threshold separating captured and non-captured exons
# Returns "True" if exon doesn't pass the filter otherwise "False"
@numba.njit
def filterZscore(mean, stdev, unCapturedThreshold):
    # Fixed paramater
    bdwthThreshold = 3  # tolerated deviation threshold

    # the mean != 0 and all samples have the same coverage value.
    # In this case a new arbitrary standard deviation is calculated
    # (simulates 5% on each side of the mean)
    if (stdev == 0):
        stdev = mean / 20

    # meanRG != 0 because of filter 1 => stdevRG != 0
    zscore = (mean - unCapturedThreshold) / stdev

    # the exon is excluded if there are less than 3 standard deviations betweenprecision
    # the threshold and the mean.
    if (zscore < bdwthThreshold):
        return True
    else:
        return False


###################
# filterSampsContribRG
# Filter n°4: samples contributing to Gaussian is less than 50%
# Given a FPM counts from an exon and a robustly fitted gaussian paramaters,
# filters the exons.
# Spec:
# - set a contribution threshold
# - obtain FPM values within +- 2 standard deviations of the mean of the Gaussian
# - calculate the contribution
# - compare the contribution to the threshold => filtering
# if the user chooses to plot the results a pdf will be created by exon
# in the folder associated with the filter
# Args:
# - exonFPM (list[floats]): FPM counts from an exon
# - mean [floats]
# - stdev [floats]
# Returns "True" if exon doesn't pass the filter otherwise "False"
@numba.njit
def filterSampsContribRG(exonFPM, mean, stdev):
    # Fixed parameters
    contributionThreshold = 0.5
    stdevLim = 2

    # targetData length is the sample number contributing to the Gaussian
    targetData = exonFPM[(exonFPM > (mean - (stdevLim * stdev))) & (exonFPM < (mean + (stdevLim * stdev))), ]

    weight = len(targetData) / len(exonFPM)

    if (weight < contributionThreshold):
        return True
    else:
        return False


#############################################################
# sampFPM2Probs
# Given a sample FPM value for an exon and parameters of an exponential and a Gaussian distribution,
# calculation of logOdds ratio for each type of copy number
# Spec:
# - calculating the likelihood probability (PDF) for the sample for each distribution (x4)
# if the user chooses to plot the results a pdf will be created by exon
# in the folder associated to exon pass filters for a selected sample
# Args:
# - mean [floats]
# - stdev [floats]
# - exponParams (list[floats]): exponential distribution paramaters (loc, scale)
# - unCapturedThreshold [floats]: FPM threshold separating captured and non-captured exons
# - exSampFPM [float]: FPM value for a sample and an exon
# Returns:
# - likelihoods (list[float]): densities for each copy number (CN0,CN1,CN2,CN3+)
def sampFPM2Probs(mean, stdev, exponParams, unCaptThreshold, exSampFPM):
    # CN2 mean shift to get CN1 mean
    meanCN1 = mean / 2

    # To Fill
    # empty list to store the densities for each copy number typeprecision
    likelihoods = np.zeros(4, dtype=np.float32)

    # Calculate the density for the exponential distribution (CN0 profil)
    # exponential distribution has a heavy tail, density calculated from it can override
    # the other Gaussian distributions.
    # associate a 0 pdf value if the sample FPM value is higher than the CN1 mean
    cdf_cno_threshold = scipy.stats.expon.cdf(unCaptThreshold, *exponParams)
    if exSampFPM <= meanCN1:
        likelihoods[0] = (1 / (1 - cdf_cno_threshold)) * scipy.stats.expon.pdf(exSampFPM, *exponParams)

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
