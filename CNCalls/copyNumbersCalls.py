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
# CNCalls:
# Given a count array and dictionaries summarising cluster information.
# Obtaining observation probabilities for each copy number type (CN0, CN1, CN2, CN3+)
# for each sample per exon.
# Arguments:
# - CNcallsArray (np.ndarray[float): probabilities, dim = NbExons x (NbSamples*4), initially -1
# - callsFilled (np.ndarray[bool]): samples filled from prevCallsArray, dim = NbSamples
# - exonsFPM (np.ndarray[float]): normalised counts from exons
# - intergenicsFPM (np.ndarray[float]): normalised counts from intergenic windows
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# - clusts2Samps (dict[str, List[int]]): key: clusterID , value: samples index list
# - clusts2Ctrls (dict[str, List[str]]): key: clusterID, value: controlsID list
# - gonoClustIDStart [str]: first clusterID from gonosomes clustering results
# - priors (list[float]): prior probability for each copy number type in the order [CN0, CN1,CN2,CN3+].
# - plotDir (str): subdir (created if needed) where result plots files will be produced
# Returns:
# - CNcallsArray (np.ndarray[float]): contains probabilities. dim=NbExons * (NbSamples*[CN0,CN1,CN2,CN3+])
def CNCalls(CNcallsArray, samples, clustID, sampsInClusts, ctrlsInClusts, specClusts, exonsFPM, intergenicsFPM, exons, priors, plotDir):

    # for DEV: creation of folder and matplotlib object for storing plots for monitoring.
    if plotDir:
        PDF_Files = []
        PDFPaths = ["F1_median.pdf", "F2_RGnotFit.pdf", "F3_zscore.pdf",
                    "F4_weight.pdf", "PASS_HOMODEL.pdf", "PASS_HETDEL.pdf",
                    "PASS_DUP.pdf", "PASS.pdf", "filteringRes_PieChart.pdf"]
        for path in PDFPaths:
            PDF_Files.append(matplotlib.backends.backend_pdf.PdfPages(os.path.join(plotDir, path)))

    # identifying autosomes and gonosomes "exons" index
    # to make calls on associated reference groups.
    maskGExIndexes = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)

    # Retrieval of cluster-specific data, all samples in the case of a presence of a control cluster,
    # and indexes of the genomic regions that need to be analyzed
    (allSampsInClust, exIndToProcess) = CNCalls.copyNumbersCalls.extractClustDimCounts(clustID, sampsInClusts, ctrlsInClusts, specClusts, maskGExIndexes)
    exonsFPMClust = exonsFPM[exIndToProcess][:, allSampsInClust]
    intergenicsFPMClust = intergenicsFPM[:, allSampsInClust]

    ###########
    # Initialize a hash allowing to detail the filtering carried out
    # as well as the calls for all the exons.
    # It is used for the pie chart representing the filtering.
    filterCounters = dict.fromkeys(["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "exonsCalls"], 0)

    ##################################
    # smoothing coverage profil averaged by exons, fit gamma distribution.
    try:
        (expParams, UncoverThreshold) = CNCalls.copyNumbersCalls.fitExponential(intergenicsFPMClust)
    except Exception as e:
        logger.error("fitExponential failed for cluster %s : %s", clustID, repr(e))
        raise Exception("fitExponential failed")

    ##############################
    # Browse cluster-specific exons
    ##############################
    for clustExon in range(exonsFPMClust.shape[0]):
        xLists = []
        yLists = []
        plotLegs = []
        verticalLines = []
        vertLinesLegs = []
        ylim = 2

        plotTitle = "ClusterID n° {} exon:{}" .format(str(clustID), '_'.join([str(e) for e in exons[exIndToProcess[clustExon]]]))

        # Print progress every 10000 exons
        if clustExon % 10000 == 0:
            print("ClusterID n°", clustID, clustExon, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # Get count data for the exon
        exonFPM = exonsFPMClust[clustExon]
        xi = np.linspace(0, max(exonFPM), 1000)

        pdfExp = scipy.stats.expon.pdf(xi, *expParams)
        xLists.append(xi)
        yLists.append(pdfExp)
        plotLegs.append("\nexpon={:.2f}, {:.2f}".format(expParams[0], expParams[1]))

        verticalLines.append(UncoverThreshold)
        vertLinesLegs.append("UncoverThreshold={:.3f}".format(UncoverThreshold))

        ###################
        # Filter n°1: exon not covered in most samples.
        # treats several possible cases:
        # - all samples in the cluster haven't coverage for the current exon
        # - more than 2/3 of the samples have no cover.
        #   Warning: Potential presence of homodeletions. We have chosen don't
        # call them because they affect too many samples
        medianFPM = np.median(exonFPM)
        if medianFPM == 0:
            filterCounters["notCaptured"] += 1

            # plot F1
            if filterCounters["notCaptured"] % 100 == 0:
                plotTitle += "\nmedian=" + '{:.2f}'.format(medianFPM)
                figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[0])

            continue

        ###################
        # Filter n°2: robust fitting of Gaussian failed.
        # the median (consider as the mean parameter of the Gaussian) is located
        # in an area without point data.
        # in this case exon is not kept for the rest of the filtering and calling step
        try:
            RGParams = CNCalls.copyNumbersCalls.fitRobustGaussian(exonFPM)
        except Exception as e:
            if str(e) == "cannot fit":
                filterCounters["cannotFitRG"] += 1

                # plot F2
                plotTitle += "\nmedian=" + '{:.2f}'.format(medianFPM)
                figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[1])

                continue
            else:
                raise

        ########################
        # apply Filters n°3 and n°4:
        # if failedFilters(exonFPM, RGParams, UncoverThreshold, filterCounters):
        #    continue
        # setting a threshold associated with standard deviation tolerated around the lowThreshold

        sdThreshold = 3
        meanRG = RGParams[0]
        stdevRG = RGParams[1]

        pdfCN2 = scipy.stats.norm.pdf(xi, meanRG, stdevRG)
        xLists.append(xi)
        yLists.append(pdfCN2)
        plotLegs.append("\nRG CN2={:.2f}, {:.2f}".format(meanRG, stdevRG))

        ylim = 2 * max(pdfCN2)

        ###################
        # Filter n°3:
        # the mean != 0 and all samples have the same coverage value.
        # In this case a new arbitrary standard deviation is calculated
        # (simulates 5% on each side of the mean)
        if (stdevRG == 0):
            stdevRG = meanRG / 20

        # meanRG != 0 because of filter 1 => stdevRG != 0
        z_score = (meanRG - UncoverThreshold) / stdevRG

        # the exon is excluded if there are less than 3 standard deviations between
        # the threshold and the mean.
        if (z_score < sdThreshold):
            filterCounters["RGClose2LowThreshold"] += 1

            # plot F3
            if filterCounters["RGClose2LowThreshold"] % 100 == 0:
                plotTitle += "\nzscore={:.2f}".format(z_score)

                figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[2])

            continue

        ###################
        # Filter n°4:
        weight = CNCalls.copyNumbersCalls.computeWeight(exonFPM, meanRG, stdevRG)
        if (weight < 0.5):
            filterCounters["fewSampsInRG"] += 1

            # plot F3
            if filterCounters["fewSampsInRG"] % 100 == 0:
                plotTitle += "\nzscore={:.2f}".format(z_score) + "\nweight={:.2f}".format(weight)
                figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[3])

            continue

        filterCounters["exonsCalls"] += 1
        # CN2 mean shift to get CN1 mean
        meanCN1 = meanRG / 2
        pdfCN1 = scipy.stats.norm.pdf(xi, meanCN1, stdevRG)
        xLists.append(xi)
        yLists.append(pdfCN1)
        plotLegs.append("\nRG CN1={:.2f}, {:.2f}".format(meanCN1, stdevRG))

        pdfCN3 = scipy.stats.norm.pdf(xi, 3 * meanCN1, stdevRG)
        xLists.append(xi)
        yLists.append(pdfCN3)
        plotLegs.append("\nRG CN3={:.2f}, {:.2f}".format(3 * meanCN1, stdevRG))

        ###################
        # compute probabilities for each sample and each CN type
        # fill CNCallsArray
        # for i in sampsInClusts[clustID]:
        i = sampsInClusts[clustID][0]
        sampFPM = exonFPM[allSampsInClust.index(i)]
        sampIndexInCallsArray = i * 4

        probNorm = CNCalls.copyNumbersCalls.computeProbabilites(sampFPM, expParams, RGParams, priors, UncoverThreshold)

        ### Plot
        verticalLines.append(sampFPM)
        vertLinesLegs.append("sample index=" + samples[i])
        plotTitle += "\nprobs={:.2f},{:.2f},{:.2f},{:.2f}".format(probNorm[0], probNorm[1], probNorm[2], probNorm[3])

        index_maxProb = np.argmax(probNorm)

        if index_maxProb == 0:
            figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[4])
        elif index_maxProb == 1:
            figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[5])
        elif index_maxProb == 3:
            if filterCounters["exonsCalls"] % 100 == 0:
                figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[6])
        else:
            if filterCounters["exonsCalls"] % 5000 == 0:
                figures.plots.plotExonProfil(exonFPM, xLists, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, PDF_Files[7])

        for val in range(4):
            if CNcallsArray[exIndToProcess[clustExon], (sampIndexInCallsArray + val)] == -1:
                CNcallsArray[exIndToProcess[clustExon], (sampIndexInCallsArray + val)] = probNorm[val]
            else:
                logger.error('erase previous probabilities values')

    figures.plots.plotPieChart(clustID, filterCounters, PDF_Files[8])

    # close the open pdfs
    for pdf in PDF_Files:
        pdf.close()
    return(CNcallsArray)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################################################
# extractClustDimCounts
# This function takes the index/ID of a cluster and various arrays containing information
# about the clusters and FPM counts as input, and extracts the exons and
# intergenic regions indexes from counts in the given cluster, accounting for the presence
# of control clusters and gonosomal regions.
# Args:
# - clustID [int]: the index/ID of the cluster
# - sampsInClusts (list of lists[str]): each sub-list contains the sample IDs for a given cluster
# - ctrlsInClusts (list of lists[str]): each sub-list contains the IDs of control clusters for a given cluster
# - specClusts (list[int]): a list that specifies whether each cluster was derived from a clustering
# analysis on autosomes (0) or gonosomes (1)
# - maskGonoExonsInd (np.ndarray[bool]): indicates whether each exon is in a gonosomal region (True) or not (False)
# - maskGonoIntergenicsInd (np.ndarray[bool]): indicates whether each intergenic region is in a gonosomal region (True) or not (False)
# The function returns a tuple containing the following variables:
# - allSampsInClust (list[str]): a list of all sample IDs in the cluster
# - exonsIndexToProcess (list[int]): indixes of exons to process
# - intergenicIndexToProcess (list[int]): indixes of intergenic regions to process
def extractClustDimCounts(clustID, sampsInClusts, ctrlsInClusts, specClusts, maskGonoExonsInd):
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


#############################################################
# fitExponential
# Given a counts array for a specific cluster identifier, coverage profile
# smoothing deduced from FPM averages per exon.
# Fit an exponential law to the data of intergenic regions associated with
# the non-coverage profile.
# f(x, scale) = (1/scale)*exp(-x/scale)
# scale parameter is the inverse of the rate parameter (lambda) used in the
# mathematical definition of the distribution.
# It is the mean of the distribution and controls the location of the distribution on the x-axis.
# location parameter is an optional parameter that controls the location of the distribution on the x-axis.
# The exponential law is the best-fitting law to the windows created at the
# best step, out of a set of 101 continuous laws tested simultaneously on both
# the Ensembl and MANE bed.
# A threshold is calculated as the point where 99% of the cumulative distribution
# function of the fitted exponential law separates covered and uncovered regions.
# This threshold will be used in the rest of the script to filter out uninterpretable
# exons that are not covered.
# Args:
# - intergenicsFPMClust (np.ndarray[floats]): FPM array specific to a cluster
# - clustID [str]: cluster identifier
# - bandwidth [int]: KDE bandwidth value, default value should be fine
# Returns a tupple (expParams, UncoverThreshold), each variable is created here:
# - expParams [tuple of floats]: estimated parameters (x2) of the exponential distribution
# - UncoverThreshold [float]: threshold for separating covered and uncovered regions,
# which corresponds to the FPM value at 99% of the CDF
def fitExponential(intergenicsFPMClust, bandWidth=0.5):
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
    expParams = scipy.stats.expon.fit(dens)

    # Calculating the threshold in FPM equivalent to 99% of the CDF (PPF = percent point function)
    UncoverThreshold = scipy.stats.expon.ppf(0.99, *expParams)

    return (expParams, UncoverThreshold)


#############################################################
# failedFilters
# Uses an exon coverage profile to identify if the exon is usable for the call.
# Filter n°3: Gaussian is too close to lowThreshold.
# Filter n°4: samples contributing to Gaussian is less than 50%.
# Args:
# - exonFPM (ndarray[float]): counts for one exon
# - RGParams (list[float]): mean and stdev from robust fitting of Gaussian
# - lowThreshold (float): FPM threshold for filter 3
# - filterCounters (dict[str:int]): key: type of filter, value: number of filtered exons
# Returns "True" if exon doesn't pass the two filters otherwise "False"
def failedFilters(exonFPM, RGParams, lowThreshold, filterCounters):
    # setting a threshold associated with standard deviation tolerated around the lowThreshold
    sdThreshold = 3
    meanRG = RGParams[0]
    stdevRG = RGParams[1]
    ###################
    # Filter n°3:
    # the mean != 0 and all samples have the same coverage value.
    # In this case a new arbitrary standard deviation is calculated
    # (simulates 5% on each side of the mean)
    if (stdevRG == 0):
        stdevRG = meanRG / 20

    # meanRG != 0 because of filter 1 => stdevRG != 0
    z_score = (meanRG - lowThreshold) / stdevRG

    # the exon is excluded if there are less than 3 standard deviations between
    # the threshold and the mean.
    if (z_score < sdThreshold):
        filterCounters["RGClose2LowThreshold"] += 1
        return(True)

    ###################
    # Filter n°4:
    weight = computeWeight(exonFPM, meanRG, stdevRG)
    if (weight < 0.5):
        filterCounters["fewSampsInRG"] += 1
        return(True)

    return(False)


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
# ancillary function of the robustGaussianFitPrivate function
# computes Gauss error function
# The error function (erf) is used to describe the Gaussian or Normal distribution.
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


#############################################################
# computeWeight
# compute the sample contribution to the Gaussian obtained in a robust way.
# Args:
# - fpm_in_exon (np.ndarray[float]): FPM values for a particular exon for each sample
# - mean (float): mean FPM value for the exon
# - standard_deviation (float): std FPM value for the exon
# Returns weight of sample contribution to the gaussian for the exon [float]
@numba.njit
def computeWeight(fpm_in_exon, mean, standard_deviation):
    targetData = fpm_in_exon[(fpm_in_exon > (mean - (2 * standard_deviation))) &
                             (fpm_in_exon < (mean + (2 * standard_deviation))), ]
    weight = len(targetData) / len(fpm_in_exon)

    return weight


#############################################################
# computeProbabilites
# Given a coverage value for a sample and distribution parameters of a gamma
# and a Gaussian for an exon, calculation of copy number observation probabilities.
# Args:
# - sampFPM (float): FPM value for a sample
# - gammaParams (list(float)): estimated parameters of the gamma distribution [shape, loc, scale]
# - RGParams (list[float]): contains mean value and standard deviation for the normal distribution
# - priors (list[float]): prior probabilities for different cases
# - lowThreshold (float):
# Returns:
# - probDensPriors (np.ndarray[float]): p(i|Ci)p(Ci) for each copy number (CN0,CN1,CN2,CN3+)
def computeProbabilites(sampFPM, expParams, RGParams, priors, UncoverThreshold):
    mean = RGParams[0]
    standard_deviation = RGParams[1]

    # CN2 mean shift to get CN1 mean
    meanCN1 = mean / 2

    # To Fill
    # Initialize an empty numpy array to store the  densities for each copy number type
    probDensities = np.zeros(4)

    ###############
    # Calculate the  density for the gamma distribution (CN0 profil)
    # This is a special case because the gamma distribution has a heavy tail,
    # which means that the density calculated from it can override
    # the other Gaussian distributions.
    # A condition is set up to directly associate a value of pdf to 0 if the sample FPM value
    # is higher than the mean of the Gaussian associated to CN1.
    # Reversely, the value of the pdf is truncated from the threshold value discriminating
    # covered from uncovered exons.
    cdf_cno_threshold = scipy.stats.gamma.cdf(UncoverThreshold, *expParams)
    if sampFPM <= meanCN1:
        probDensities[0] = (1 / (1 - cdf_cno_threshold)) * scipy.stats.gamma.pdf(sampFPM, *RGParams)

    ################
    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    probDensities[1] = scipy.stats.norm.pdf(sampFPM, meanCN1, standard_deviation)
    probDensities[2] = scipy.stats.norm.pdf(sampFPM, mean, standard_deviation)
    probDensities[3] = scipy.stats.norm.pdf(sampFPM, 3 * meanCN1, standard_deviation)

    #################
    # Add prior probabilities
    probDensPriors = np.multiply(probDensities, priors)

    # normalized probabilities
    # probNorm = probDensPriors / np.sum(probDensPriors)

    return probDensPriors

    ######## EARLY RETURN, for dev step4
    ################
    # case where one of the probabilities is equal to 0 addition of an epsilon
    # which is 1000 times lower than the lowest probability
    probability_densities_priors = addEpsilonPrivate(probability_densities_priors)

    ##################
    # Calculate the log-odds ratios
    emissionProba = np.zeros(4)
    for i in range(len(probability_densities_priors)):
        # Calculate the denominator for the log-odds ratio
        to_subtract = np.sum(probability_densities_priors[np.arange(probability_densities_priors.shape[0]) != i])

        # Calculate the log-odds ratio for the current probability density
        log_odd = np.log10(probability_densities_priors[i]) - np.log10(to_subtract)
        # probability transformation
        emissionProba[i] = 1 / (1 + np.exp(log_odd))

    return emissionProba / emissionProba.sum()  # normalized


# #############################################################
# # addEpsilon
# @numba.njit
# def addEpsilon(probs, epsilon_factor=1000):
#     min_prob = np.min(probs[probs > 0])
#     epsilon = min_prob / epsilon_factor
#     probs = np.where(probs == 0, epsilon, probs)
#     return probs
