import logging
import os
import numpy as np
import numba
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
#####################################
# getData2Process
# Given index/ID of a cluster, various arrays containing information about the clusters
# and a boolean mask on the exonsNB according to their presence on autosomes(0) or gonosomes(1),
# get the complete samples list (with samples from reference cluster) and the exons indexes
# to use for analyse cluster
# Args:
# - clustID [int]: the index/ID of the cluster
# - sampsInClusts (list of lists[str]): each sub-list contains the sample IDs for a given cluster
# - ctrlsInClusts (list of lists[str]): each sub-list contains the IDs of control clusters for a given cluster
# - specClusts (list[int]): a list that specifies whether each cluster was derived from a clustering
# analysis on autosomes (0) or gonosomes (1)
# - maskGonoExonsInd (np.ndarray[bool]): indicates whether each exon is in a gonosomal region (True) or not (False)
# The function returns a tuple containing the following variables:
# - allSampsInClust (list[str]): a list of all sample IDs in the cluster
# - exonsIndexToProcess (list[int]): indixes of exons to process
def getData2Process(clustID, sampsInClusts, ctrlsInClusts, specClusts, maskGonoExonsInd):
    # To fill and returns
    allSampsInClust = []
    exonsIndexToProcess = []

    # Get specific cluster samples
    allSampsInClust.extend(sampsInClusts[clustID].copy())

    # If there are control clusters, add their samples to the list of all samples in the cluster
    if ctrlsInClusts[clustID]:
        for ctrlID in ctrlsInClusts[clustID]:
            allSampsInClust.extend(sampsInClusts[ctrlID])

    # Check if the cluster comes from autosomes or gonosomes, and extract exons/intergenic regions indexes
    if specClusts[clustID] == 0:
        exonsIndexToProcess.extend(np.where(~maskGonoExonsInd)[0].tolist())
    else:
        exonsIndexToProcess.extend(np.nonzero(~maskGonoExonsInd)[0].tolist())

    return(allSampsInClust, exonsIndexToProcess)


#####################################
# CNCalls
# Given a cluster identifier, two arrays containing FPM values for pseudo-exons and exons,
# as well as lists of definitions for samples and clusters, this function calculates the
# log-odds for each copy number type (CN0, CN1, CN2, CN3+) for each exon and each sample.
# Specifically:
# - fit an exponential law from all pseudo exons => non-capture profile, obtain the
# law parameters and a FPM threshold corresponding to 99% of the cdf.
# - filtering of non-interpretable exons and fitting a robust Gaussian:
#    - F1: not captured => median coverage of the exon = 0
#    - F2: cannotFitRG => robustly impossible to fit a Gaussian (median close to 0)
#    - F3: RGClose2UncovThreshold => fitted Gaussian overlaps the threshold associated
#          with the uncaptured exon profile
#    - F4: fewSampsInRG => the sample contribution rate to the robust Gaussian is too low (<50%)
# - log Odds calculation (Bayes' theorem) for each sample/exon,
#    - CN0: exponential (case where the sample FPM value is lower than the CN1 mean otherwise 0)
#    - CN1, CN2, CN3: robust Gaussian fit
# - filtered and unfiltered exon profiles and pie charts representations are produced
# in DEBUG mode, saved in different PDF files.
# Args:
# - clustID [int]
# - exonsFPM (np.ndarray[float]): normalised counts from exons
# - intergenicsFPM (np.ndarray[float]): normalised counts from intergenic windows
# - samples (list [str]): samples names
# - sampsInClusts (list[int]): "samples" indexes list constituting the cluster, specific samples
#   cluster and samples from reference cluster
# - sampsSpe (list[int]): "samples" indexes list constituting the cluster, only specific samples cluster
# - exonsToProcess (list of lists[str,int,int,str]): information on exons to be analyzed,
#   either autosomes or gonosomes, containing CHR,START,END,EXON_ID
# - priors (list[float]): prior probability for each copy number type in the order [CN0, CN1,CN2,CN3+]
# - plotFolder (str): subdir (created if needed) where result plots files will be produced
# Returns a tupple (clustCalls, exonsCalls), each variable are created here:
# - clustCalls (list of lists[floats]): index in the main list is a sample where each sub-list corresponds
# to the log-odds for the 4 copy numbers for interpretable exons
# - exonsCallsStatus (list[int]): status for index exons that passed the filters = 1,
# not pass filters F1=-1, F2=-2, F3=-3, F4=-4, initiate to 0.
def CNCalls(clustID, exonsFPM, intergenicsFPM, samples, sampsInClusts, sampsSpe, exonsToProcess, priors, plotFolders=None):
    # To Fill and returns
    clustCalls = [[] for _ in range(len(sampsSpe))]
    exonsCallsStatus = [""] * len(exonsToProcess)
    # cluster-specific data
    exonsFPMClust = exonsFPM[exonsToProcess][:, sampsInClusts]
    intergenicsFPMClust = intergenicsFPM[:, sampsInClusts]

    # fit an exponential distribution from all pseudo exons
    try:
        (expParams, unCaptThreshold) = fitExponential(intergenicsFPMClust)
        logger.info("clustID : %i, exponential params loc= %.4f scale=%.4f, uncaptured threshold = %.4fFPM", clustID, expParams[0], expParams[1], unCaptThreshold)
    except Exception as e:
        logger.error("fitExponential failed for cluster %i : %s", clustID, repr(e))
        raise Exception("fitExponential failed")

    # Browse cluster-specific exons
    for exonIndex in range(exonsFPMClust.shape[0]):
        # Print progress every 10000 exons
        if exonIndex % 10000 == 0:
            print("ClusterID n°", clustID, exonIndex, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # Get count data for the exon
        exonFPM = exonsFPMClust[exonIndex]

        # Filter n°1: not captured => median coverage of the exon = 0
        if FilterUncapturedExons(exonFPM, plotFolders, clustID, exonsToProcess[exonIndex], expParams, unCaptThreshold):
            exonsCallsStatus[exonIndex] = "notCaptured"
            continue
        # robustly fit a gaussian
        # Filter n°2: impossible (median close to 0)
        try:
            RGParams = fitRobustGaussian(exonFPM)
        except Exception as e:
            if str(e) == "cannot fit":
                exonsCallsStatus[exonIndex] = "cannotFitRG"
                continue
            else:
                raise e
        # Filter n°3: RG overlaps the threshold associated with the uncaptured exon profile
        if FilterZscore(RGParams, unCaptThreshold, plotFolders, exonFPM, clustID, exonsToProcess[exonIndex], expParams):
            exonsCallsStatus[exonIndex] = "RGClose2LowThreshold"
            continue
        # Filter n°4: the sample contribution rate to the robust Gaussian is too low (<50%)
        if FilterSampsContribRG(exonFPM, RGParams):
            exonsCallsStatus[exonIndex] = "fewSampsInRG"
            continue

        exonsCallsStatus[exonIndex] = "exonsCalls"

        # log Odds calculation for each sample/exon
        for i in range(len(sampsInClusts)):
            if sampsInClusts[i] in sampsSpe:
                sampFPM = exonFPM[i]
                sampIndexInOutput = sampsSpe.index(sampsInClusts[i])

                probNorm = computeLogOdds([samples[sampsInClusts[i]], sampFPM],
                                          expParams, RGParams, priors, unCaptThreshold,
                                          plotFolders, clustID, exonFPM, exonsToProcess[exonIndex])

                clustCalls[sampIndexInOutput].append(probNorm)
            else:
                continue

    if plotFolders:
        counts = {}
        for value in exonsCallsStatus:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
        
        figures.plots.plotPieChart(clustID, exonsCallsStatus, plotFolders[-1])

    return (clustCalls, exonsCallsStatus)


###########################################
# populateResArray
# Given copyNumber call results for a cluster for specific exon indices, filling the results array.
# Spec:
# - browse the indices of the exons where a call was made
# - find the indices of the columns corresponding to the samples
# - copy the data into the array
# Args:
# - clustCalls (list of lists[floats]): index in the main list is a sample where each sub-list corresponds
# to the log-odds for the 4 copy numbers for interpretable exons
# - exonsCallsStatus (list[int]): status for index exons that passed the filters = 1,
# not pass filters F1=-1, F2=-2, F3=-3, F4=-4.
# - samples (list[str]): samples names
# - sampsSpeClust (list[int]): "samples" indexes list for a clusterID
# - exIndToProcess (list[int]): exons indexes to analyse for a clusterID
# - CNcallsArray (np.array[int]): logOdds 
def populateResArray(clustCalls, exonsCallsStatus, samples, sampsSpeClust, exIndToProcess, CNcallsArray):
    for exonInd in range(len(exIndToProcess)):
        if exonsCallsStatus[exonInd] == 1:
            for sampInd in range(len(sampsSpeClust)):
                sampIndexInCallsArray = samples.index(sampsSpeClust[sampInd]) * 4

                for val in range(4):
                    if CNcallsArray[exIndToProcess[exonInd], (sampIndexInCallsArray + val)] == -1:
                        CNcallsArray[exIndToProcess[exonInd], (sampIndexInCallsArray + val)] = clustCalls[sampInd][exonInd]
                    else:
                        raise Exception('erase previous probabilities values')

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
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
# - clustID [str]: cluster identifier
# - bandwidth [int]: KDE bandwidth value, default value should be fine
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

    # Fitting the exponential distribution and retrieving associated parameters => tupple(scale,loc)
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
# if the user chooses to plot the results a pdf will be created by exon
# in the folder associated with the filter
# Args:
# - exonFPM (ndarray[float]): counts for one exon
# - plotFolders (list [str]): path to the folders where to save the pdfs
# - clustID [int]
# - exonsList (list[str,int,int,str]): infos for one exon [CHR,START,END,EXONID]
# - expParams (list[float]): loc and scale from exponential fitting
# - unCaptThreshold (float): FPM threshold separating uncaptured exons from captured exons
# Returns "True" if exon doesn't pass the filter otherwise "False"
def FilterUncapturedExons(exonFPM, plotFolders, clustID, exonsList, expParams, unCaptThreshold):
    medianFPM = np.median(exonFPM)
    if medianFPM == 0:
        if plotFolders:
            preprocessPlotData(clustID, exonFPM, exonsList, expParams, plotFolders[0], unCaptThreshold)
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
# FilterZscore
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
# if the user chooses to plot the results a pdf will be created by exon
# in the folder associated with the filter
# Args:
# - RGParams (list[float]): mean and stdev from robust fitting of Gaussian
# - unCaptThreshold (float): FPM threshold
# - plotFolders (list [str]): path to the folders where to save the pdfs
# - exonFPM (ndarray[float]): counts for one exon
# - clustID [int]
# - exonsList (list[str,int,int,str]): infos for one exon [CHR,START,END,EXONID]
# - expParams (list[float]): loc and scale from exponential fitting
# Returns "True" if exon doesn't pass the filter otherwise "False"
def FilterZscore(RGParams, unCaptThreshold, plotFolders, exonFPM, clustID, exonsList, expParams):
    # Fixed paramater
    bdwthThreshold = 3

    meanRG = RGParams[0]
    stdevRG = RGParams[1]

    # the mean != 0 and all samples have the same coverage value.
    # In this case a new arbitrary standard deviation is calculated
    # (simulates 5% on each side of the mean)
    if (stdevRG == 0):
        stdevRG = meanRG / 20

    # meanRG != 0 because of filter 1 => stdevRG != 0
    zscore = (meanRG - unCaptThreshold) / stdevRG

    # the exon is excluded if there are less than 3 standard deviations between
    # the threshold and the mean.
    if (zscore < bdwthThreshold):
        if plotFolders:
            preprocessPlotData(clustID, exonFPM, exonsList, expParams, plotFolders[1], unCaptThreshold, RGParams=RGParams, zscore=zscore)
        return True
    else:
        return False


###################
# FilterSampsContribRG
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
# - exonFPM (ndarray[float]): counts for one exon
# - RGParams (list[float]): mean and stdev from robust fitting of Gaussian
# - plotFolders (list [str]): path to the folders where to save the pdfs
# - unCaptThreshold (float): FPM threshold
# - clustID [int]
# - exonsList (list[str,int,int,str]): infos for one exon [CHR,START,END,EXONID]
# - expParams (list[float]): loc and scale from exponential fitting
# Returns "True" if exon doesn't pass the filter otherwise "False"
def FilterSampsContribRG(exonFPM, RGParams, plotFolders, unCaptThreshold, clustID, exonsList, expParams):
    # Fixed parameter
    contributionThreshold = 0.5

    meanRG = RGParams[0]
    stdevRG = RGParams[1]

    # targetData length is the sample number contributing to the Gaussian
    targetData = exonFPM[(exonFPM > (meanRG - (2 * stdevRG))) & (exonFPM < (meanRG + (2 * stdevRG))), ]

    weight = len(targetData) / len(exonFPM)

    if (weight < contributionThreshold):
        if plotFolders:
            preprocessPlotData(clustID, exonFPM, exonsList, expParams, plotFolders[2], unCaptThreshold, RGParams=RGParams, weight=weight)
        return True
    else:
        return False


#############################################################
# computeLogOdds
# Given a sample FPM value for an exon and parameters of an exponential and a Gaussian distribution,
# calculation of logOdds ratio for each type of copy number
# Spec:
# - calculating the likelihood probability (PDF) for the sample for each distribution (x4)
# - addition of priors
# - add an epsilon if probability to avoid any zero-valued probabilities
# - logOdds calculation via Bayes' theory
# if the user chooses to plot the results a pdf will be created by exon
# in the folder associated to exon pass filters for a selected sample
# Args:
# - sampInfos (list[str,float, int]): sample name + FPM value
# - expParams (list(float)): estimated parameters of the exponential distribution [loc, scale]
# - RGParams (list[float]): estimated parameters of the normal distribution [mean, stdev]
# - priors (list[float]): prior probabilities for different cases
# - unCaptThreshold [float]: FPM threshold
# - plotFolders (list [str]): path to the folders where to save the pdfs
# - clustID [int]
# - exonFPM (ndarray[float]): counts for one exon
# - exonsList (list[str,int,int,str]): infos for one exon [CHR,START,END,EXONID]
# Returns:
# - LogOdds (list[float]): p(i|Ci)p(Ci) for each copy number (CN0,CN1,CN2,CN3+)
def computeLogOdds(sampInfos, expParams, RGParams, priors, unCaptThreshold, plotFolders, clustID, exonFPM, exonsList):
    mean = RGParams[0]
    stdev = RGParams[1]

    # CN2 mean shift to get CN1 mean
    meanCN1 = mean / 2

    # To Fill
    # empty list to store the densities for each copy number type
    probDensities = [0] * 4
    # empty list to store logOdds
    LogOdds = []

    # Calculate the density for the exponential distribution (CN0 profil)
    # This is a special case because the exponential distribution has a heavy tail,
    # which means that the density calculated from it can override
    # the other Gaussian distributions.
    # A condition is set up to directly associate a value of pdf to 0 if the sample FPM value
    # is higher than the mean of the Gaussian associated to CN1.
    # Reversely, the value of the pdf is truncated from the threshold value discriminating
    # covered from uncovered exons.
    cdf_cno_threshold = scipy.stats.expon.cdf(unCaptThreshold, *expParams)
    if sampInfos[1] <= meanCN1:
        probDensities[0] = (1 / (1 - cdf_cno_threshold)) * scipy.stats.expon.pdf(sampInfos[1], *expParams)

    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    probDensities[1] = scipy.stats.norm.pdf(sampInfos[1], meanCN1, stdev)
    probDensities[2] = scipy.stats.norm.pdf(sampInfos[1], mean, stdev)
    probDensities[3] = scipy.stats.norm.pdf(sampInfos[1], 3 * meanCN1, stdev)

    # Add prior probabilities
    probDensPriors = [a * b for a, b in zip(probDensities, priors)]

    # case where one of the probabilities is equal to 0 addition of an epsilon
    # which is 1000 times lower than the lowest probability
    probDensPriors = addEpsilon(probDensPriors)

    # log-odds ratios calculation
    for i in range(len(probDensPriors)):
        # denominator for the log-odds ratio
        toSubtract = np.sum(probDensPriors[np.arange(probDensPriors.shape[0]) != i])

        # log-odds ratio for the current probability density
        logOdd = np.log10(probDensPriors[i]) - np.log10(toSubtract)

        LogOdds.append(logOdd)

    if plotFolders:
        preprocessPlotData(clustID, exonFPM, exonsList, expParams, plotFolders[3], unCaptThreshold, RGParams=RGParams, sampInfos=sampInfos.extend(LogOdds))

    return LogOdds


#############################################################
# addEpsilon
# Add a small epsilon value to the probability distribution `probs`, to avoid any zero-valued probabilities.
# Args:
# - probs (list[float]): probability distribution.
# - epsilon_factor (int): factor that determines the magnitude of the added epsilon value.
# Returns:
# -probs (list[float]): The modified probability distribution with added epsilon values.
@numba.njit
def addEpsilon(probs, epsilon_factor=1000):
    # Find the smallest non-zero probability value
    min_prob = min(filter(lambda x: x > 0, probs))
    # Compute the epsilon value by dividing the smallest non-zero probability by the epsilon factor
    epsilon = min_prob / epsilon_factor
    # Replace any zero-valued probabilities with the computed epsilon value
    probs = [epsilon if p == 0 else p for p in probs]
    return probs


#########################
# preprocessPlotData
# create a plot of the exon FPM values and the probability density functions and vertical lines,
# and saves the resulting plot as a PDF in the specified folder.
# Args:
# - clustID [int]
# - exonFPM (np.ndarray[floats]): FPM values for the exon in the cluster
# - exonsList (list[str, int, int, str]): a list of integers representing the exons in the cluster
# - expParams (list[float]): estimated parameters of the exponential distribution [loc, scale]
# - folder [str]: path to a folder where to save the pdfs
# - unCaptThreshold [float]: FPM threshold
# - median [float]: (optional) median value from exon FPM values
# - RGParams (list[float]): (optional) estimated parameters of the gaussian distribution [mean, stdev]
# - zscore [float]: (optional)
# - weight [float]: (optional)
# - sampInfos (list[str,float, float, float, float, float]): (optional) sample name, sample FPM value,
# and probabilities for CN1, CN2, and CN3
def preprocessPlotData(clustID, exonFPM, exonsList, expParams, folder, unCaptThreshold, median=None, RGParams=None, zscore=None, weight=None, sampInfos=None):
    # initiate plots variables
    figTitle = "ClusterID_{}_{}" .format(str(clustID), '_'.join([str(e) for e in exonsList]))
    plotTitle = "ClusterID n° {} exon:{}" .format(str(clustID), '_'.join([str(e) for e in exonsList]))
    yLists = []
    plotLegs = []
    verticalLines = []
    vertLinesLegs = []
    ylim = 10

    # creates FPM ranges base
    xi = np.linspace(0, max(exonFPM), 1000)

    # calculate the probability density function for the exponential distribution
    pdfExp = scipy.stats.expon.pdf(xi, *expParams)
    yLists.append(pdfExp)
    plotLegs.append("\nexpon={:.2f}, {:.2f}".format(expParams[0], expParams[1]))

    verticalLines.append(unCaptThreshold)
    vertLinesLegs.append("UncoverThreshold={:.3f}".format(unCaptThreshold))

    # populate the plot variablesaccording to the presence of the arguments passed to the function
    if median:
        plotTitle += "\nmedian=" + '{:.2f}'.format(median)
        figTitle = '{:.2f}'.format(median) + "_" + figTitle

    if RGParams:
        meanRG = RGParams[0]
        stdevRG = RGParams[1]

        pdfCN2 = scipy.stats.norm.pdf(xi, meanRG, stdevRG)
        yLists.append(pdfCN2)
        plotLegs.append("\nRG CN2={:.2f}, {:.2f}".format(meanRG, stdevRG))

        ylim = 2 * max(pdfCN2)

    if zscore:
        plotTitle += "\nzscore={:.2f}".format(zscore)
        figTitle = '{:.2f}'.format(zscore) + "_" + figTitle

    if weight:
        plotTitle += "\nweight={:.2f}".format(weight)
        figTitle = '{:.2f}'.format(weight) + "_" + figTitle

    if sampInfos:
        meanCN1 = meanRG / 2
        pdfCN1 = scipy.stats.norm.pdf(xi, meanCN1, stdevRG)
        yLists.append(pdfCN1)
        plotLegs.append("\nRG CN1={:.2f}, {:.2f}".format(meanCN1, stdevRG))

        pdfCN3 = scipy.stats.norm.pdf(xi, 3 * meanCN1, stdevRG)
        yLists.append(pdfCN3)
        plotLegs.append("\nRG CN3={:.2f}, {:.2f}".format(3 * meanCN1, stdevRG))

        verticalLines.append(sampInfos[1])
        vertLinesLegs.append("sample name=" + sampInfos[0])
        plotTitle += "\nprobs={:.2f}, {:.2f}, {:.2f}, {:.2f}".format(sampInfos[2], sampInfos[3], sampInfos[4], sampInfos[5])
        index_maxProb = np.argmax(sampInfos[2:5])

        figTitle = 'CN{:.i}'.format(index_maxProb - 2) + "_" + sampInfos[0] + "_" + figTitle

    PDF_File = matplotlib.backends.backend_pdf.PdfPages(os.path.join(folder, figTitle))
    figures.plots.plotExonProfil(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_File)
    PDF_File.close()
