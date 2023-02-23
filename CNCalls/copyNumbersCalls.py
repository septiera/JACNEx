import logging
import os
import gzip
import numpy as np
import numba
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time

import clusterSamps.smoothing
import clusterSamps.genderDiscrimination


# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
# - countsFPM (np.ndarray[float]): normalised counts
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# - sex2Clust dict[str, list[str]]: key: "A" autosomes or "G" gonosome, value: clusterID list
# - clusts2Samps (dict[str, List[int]]): key: clusterID , value: samples index list based on SOIs list
# - clusts2Ctrls (dict[str, List[str]]): key: clusterID, value: controlsID list
# - priors (list[float]): prior probability for each copy number type in the order [CN0, CN1,CN2,CN3+].
# - SOIs (list[str]): sampleIDs list
# - plotDir (str): subdir (created if needed) where result plots files will be produced
# Returns:
# - emissionArray (np.ndarray[float]): contain probabilities. dim=NbExons * (NbSOIs*[CN0,CN1,CN2,CN3+])
def CNCalls(countsFPM, exons, sex2Clust, clusts2Samps, clusts2Ctrls, priors, SOIs, plotDir):
    emissionArray = allocateProbsArray(len(SOIs), len(exons))

    # create a matplotlib object and open a pdf
    pdfFile = os.path.join(plotDir, "ResCNCallsByCluster_" + str(len(SOIs)) + "samps.pdf")
    PDF = matplotlib.backends.backend_pdf.PdfPages(pdfFile)

    # when discriminating between genders,
    # importance of identifying autosomes and gonosomes "exons" index
    # to make calls on associated reference groups.
    if sex2Clust:
        gonoIndex, _ = clusterSamps.genderDiscrimination.getGenderInfos(exons)
        maskAutosome_Gonosome = ~np.isin(np.arange(countsFPM.shape[0]), sorted(set(sum(gonoIndex.values(), []))))

    ##############################
    # Browse clusters
    ##############################
    for clustID in clusts2Samps:
        # retrieve samples and exons 
        SOIsIndex = CNCalls.copyNumbersCalls.extractSamps(clustID, clusts2Samps, clusts2Ctrls)
        logger.info("Cluster %s, nb SOIs sortie fonction %s", clustID, len(SOIsIndex))

        if sex2Clust:
            exonsIndex = CNCalls.copyNumbersCalls.extractExons(clustID, sex2Clust, maskAutosome_Gonosome)
        else:
            exonsIndex = range(len(exons))

        # Create Boolean masks for columns and rows
        col_mask = np.isin(np.arange(countsFPM.shape[1]), SOIsIndex, invert=True)
        row_mask = np.isin(np.arange(countsFPM.shape[0]), exonsIndex, invert=True)

        # Use the masks to index the 2D numpy array
        clusterCounts = countsFPM[np.ix_(row_mask, col_mask)]

        ###########
        # Initialize  a hash allowing to detail the filtering carried out
        # as well as the calls for all the exons.
        # It is used for the pie chart representing the filtering.
        filterCounters = dict.fromkeys(["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "exonsCalls"], 0)

        ##################################
        # smoothing coverage profil averaged by exons, fit gamma distribution.
        gammaParams, lowThreshold = fitGammaDistribution(clusterCounts, clustID, PDF)

        ##############################
        # Browse cluster-specific exons
        ##############################
        for clustExon in range(clusterCounts.shape[0]):
            # Print progress every 10000 exons
            if clustExon % 10000 == 0:
                logger.info("ClusterID %s: %s  %s ", clustID, clustExon, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

            # Get count data for the exon
            exonFPM = clusterCounts[clustExon]

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
                continue

            ###################
            # Filter n°2: robust fitting of Gaussian failed.
            # the median (consider as the mean parameter of the Gaussian) is located
            # in an area without point data.
            # in this case exon is not kept for the rest of the filtering and calling step
            try:
                RGParams = fitRobustGaussian(exonFPM)
            except Exception as e:
                if str(e) == "cannot fit":
                    filterCounters["cannotFitRG"] += 1
                    continue
                else:
                    raise

            ########################
            # apply Filters n°3 and n°4:
            if failedFilters(exonFPM, RGParams, lowThreshold, filterCounters):
                continue

            filterCounters["exonsCalls"] += 1

            ###################
            # Retrieve results for each sample
            for i in clusts2Samps[clustID]:
                sampFPM = exonFPM[SOIsIndex.index(i)]
                sampIndexInEmissionArray = i * 4

                probNorm = computeProbabilites(sampFPM, gammaParams, RGParams, priors, lowThreshold)

                for val in range(4):
                    if emissionArray[exonsIndex[clustExon], (sampIndexInEmissionArray + val)] == -1:
                        emissionArray[exonsIndex[clustExon], (sampIndexInEmissionArray + val)] = probNorm[val]
                    else:
                        logger.error('erase previous probabilities values')

        filtersPiePlot(clustID, filterCounters, PDF)

    # close the open pdf
    PDF.close()
    return(emissionArray)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
# extractSamps
# Given a cluster identifier and dictionaries reporting cluster information
# Extraction SOIs indexes
# Args:
# - clustID [str] : cluster identifier
# - clusts2Samps (dict[str: list[int]]): key:cluster identifier, value: SOIs indexes list
# - clusts2Ctrls (dict[str: list[str]]): key:cluster identifier, value: list of control clusters
# Returns:
# - SOIsIndex (list[int]): SOIs indexes in current cluster
def extractSamps(clustID, clusts2Samps, clusts2Ctrls):
    ##### COLUMN indexes in countsFPM => samples
    # Get the indexes of the samples in the cluster and its controls
    SOIsIndex = clusts2Samps[clustID].copy()
    if clustID in clusts2Ctrls:
        for control in clusts2Ctrls[clustID]:
            SOIsIndex.extend(clusts2Samps[control].copy())
    return SOIsIndex


#############################################################
# extractExons
# Given a cluster identifier and dictionaries reporting cluster sexual information
# Extraction exons indexes (specific to autosomes or gonosomes)
# Args:
# - clustID [str] : cluster identifier
# - sex2Clust (dict[str, list[str]]): for "A" autosomes or "G" gonosomes a list of corresponding clusterIDs is associated
# - mask (numpy.ndarray[bool]): boolean mask 1: autosome exon indexes, 0: gonosome exon indexes. dim=NbExons
# Returns:
# - SOIsIndex (list[int]): SOIs indexes in current cluster
def extractExons(clustID, sex2Clust, mask):
    ##### ROW indexes in countsFPM => exons
    # in case there are specific autosome and gonosome clusters.
    # identification of the indexes of the exons associated with the gonosomes or autosomes.
    if sex2Clust:
        if clustID in sex2Clust["A"]:
            exonsIndex = np.where(mask)[0]
        else:
            exonsIndex = np.where(~mask)[0]
    else:
        exonsIndex = range(len(mask))

    return exonsIndex


#############################################################
# fitGammaDistribution
# Given a counts array for a specific cluster identifier, coverage profile
# smoothing deduced from FPM averages per exon.
# Returns the parameters of a distribution that best fits the coverage profile
# of poorly covered exons and a FPM threshold (lowThreshold).
# Exons with FPM lower than this lowThreshold are both uncaptured, poorly
# covered and potentially homodeleted.
# Args:
# - clusterCounts (np.ndarray[floats]): counts array
# - clustID (str): cluster identifier
# - PDF (matplotlib object): store plots in a single pdf
# Returns a tupple (gammaParams, lowThreshold), each variable is created here:
# - gammaParams (tuple of floats): estimated parameters of the gamma distribution
# - lowThreshold (float): value corresponding to 95% of the gamma cumulative distribution
# function
def fitGammaDistribution(clusterCounts, clustID, PDF):
    # compute meanFPM by exons
    # save computation time instead of taking the raw data (especially for clusters
    # with many samples)
    meanCountByExons = np.mean(clusterCounts, axis=1)

    # smooth the coverage profile with kernel-density estimate using Gaussian kernels
    # - binEdges (np.ndarray[floats]): FPM range from 0 to 10 every 0.1
    # - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
    #   dim= len(binEdges)
    binEdges, densityOnFPMRange = clusterSamps.smoothing.smoothingCoverageProfile(meanCountByExons)

    # recover the threshold of the minimum density means before an increase
    # - minIndex (int): index from "densityMeans" associated with the first lowest
    # observed mean
    minIndex = clusterSamps.smoothing.findLocalMin(densityOnFPMRange)[0]

    countsExonsNotCovered = meanCountByExons[meanCountByExons <= binEdges[minIndex]]

    countsExonsNotCovered.sort()  # sort data in-place

    # The gamma distribution was chosen after testing 101 continuous distribution laws,
    #  -it has few parameters (3 in total: shape, loc, scale=1/beta),
    #  -is well-known, and had the best goodness of fit on the empirical data.
    # estimate the parameters of the gamma distribution that best fits the data
    gammaParams = scipy.stats.gamma.fit(countsExonsNotCovered)

    # cumulative distribution
    cdf = scipy.stats.gamma.cdf(countsExonsNotCovered, a=gammaParams[0], loc=gammaParams[1], scale=gammaParams[2])

    # find the index of the last element where cdf < 0.95
    thresholdIndex = np.where(cdf < 0.95)[0][-1]

    lowThreshold = countsExonsNotCovered[thresholdIndex]

    coverageProfilPlot(clustID, binEdges, densityOnFPMRange, minIndex, lowThreshold, clusterCounts.shape[1], PDF)

    return (gammaParams, lowThreshold)


#############################################################
# coverageProfilPlot
# generates a plot per cluster
# x-axis: the range of FPM bins (every 0.1 between 0 and 10)
# y-axis: exons densities
# black curve: density data smoothed with kernel-density estimate using Gaussian kernels
# red vertical line: minimum FPM threshold, all uncovered exons are below this threshold
# green curve: gamma fit
#
# Args:
# - sampleName (str): sample name
# - binEdges (np.ndarray[floats]): FPM range
# - densityOnFPMRange (np.ndarray[float]): densities from binEdges
# - minIndex (int): binEdges index associated with the first lowest density observed
# - lowThreshold (float): FPM threshold
# from the gamma, corresponds to the FPM threshold where before this the exons are not covered
# (contains both uncaptured, poorly covered and potentially homodeleted exons).
# - SOIsNb (int): number of samples in the cluster
# - pdf (matplotlib object): store plots in a single pdf
# Returns and saves a plot in the output pdf
def coverageProfilPlot(clustID, binEdges, densityOnFPMRange, minIndex, lowThreshold, SOIsNb, PDF):

    fig = matplotlib.pyplot.figure(figsize=(6, 6))
    matplotlib.pyplot.plot(binEdges, densityOnFPMRange, color='black', label='smoothed densities')
    matplotlib.pyplot.axvline(binEdges[minIndex], color='crimson', linestyle='dashdot', linewidth=2,
                              label="minFPM=" + '{:0.1f}'.format(binEdges[minIndex]))
    matplotlib.pyplot.axvline(lowThreshold, color='blue', linestyle='dashdot', linewidth=2,
                              label="uncovExonThreshold=" + '{:0.2f}'.format(lowThreshold))
    matplotlib.pyplot.ylim(0, 0.5)
    matplotlib.pyplot.ylabel("Exon densities")
    matplotlib.pyplot.xlabel("Fragments Per Million")
    matplotlib.pyplot.title("ClusterID:" + clustID + " coverage profile (" + str(SOIsNb) + ")")
    matplotlib.pyplot.legend()

    PDF.savefig(fig)
    matplotlib.pyplot.close()


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
    if (z_score < 3):
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
def computeProbabilites(sampFPM, gammaParams, RGParams, priors, lowThreshold):
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
    cdf_cno_threshold = scipy.stats.gamma.cdf(lowThreshold, a=gammaParams[0], loc=gammaParams[1], scale=gammaParams[2])
    if sampFPM <= meanCN1:
        probDensities[0] = (1 / (1 - cdf_cno_threshold)) * scipy.stats.gamma.pdf(sampFPM,
                                                                                 a=gammaParams[0],
                                                                                 loc=gammaParams[1],
                                                                                 scale=gammaParams[2])

    ################
    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    probDensities[1] = scipy.stats.norm.pdf(sampFPM, meanCN1, standard_deviation)
    probDensities[2] = scipy.stats.norm.pdf(sampFPM, mean, standard_deviation)
    probDensities[3] = scipy.stats.norm.pdf(sampFPM, 3 * meanCN1, standard_deviation)

    #################
    # Add prior probabilities
    probDensPriors = np.multiply(probDensities, priors)

    # # normalized probabilities
    # probNorm = probability_densities_priors / np.sum(probability_densities_priors)

    # ################
    # # case where one of the probabilities is equal to 0 addition of an epsilon
    # # which is 1000 times lower than the lowest probability
    # probability_densities_priors = addEpsilonPrivate(probability_densities_priors)

    # ##################
    # # Calculate the log-odds ratios
    # emissionProba = np.zeros(4)
    # for i in range(len(probability_densities_priors)):
    #     # Calculate the denominator for the log-odds ratio
    #     to_subtract = np.sum(probability_densities_priors[np.arange(probability_densities_priors.shape[0]) != i])

    #     # Calculate the log-odds ratio for the current probability density
    #     log_odd = np.log10(probability_densities_priors[i]) - np.log10(to_subtract)
    #     # probability transformation
    #     emissionProba[i] = 1 / (1 + np.exp(log_odd))

    # return emissionProba / emissionProba.sum()  # normalized

    return probDensPriors


# #############################################################
# # addEpsilon
# @numba.njit
# def addEpsilon(probs, epsilon_factor=1000):
#     min_prob = np.min(probs[probs > 0])
#     epsilon = min_prob / epsilon_factor
#     probs = np.where(probs == 0, epsilon, probs)
#     return probs


#############################################################
# filtersPiePlot:
# generates a plot per cluster
# Args:
# - clustID [str]: cluster identifier
# - filterCounters (dict[str:int]): dictionary of exon counters of different filtering
# performed for the cluster
# - pdf (matplotlib object): store plots in a single pdf
#
# save a plot in the output pdf
def filtersPiePlot(clustID, filterCounters, pdf):

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    matplotlib.pyplot.pie(filterCounters.values(), labels=filterCounters.keys(),
                          colors=["grey", "yellow", "mediumpurple", "royalblue", "mediumaquamarine"],
                          autopct=lambda x: str(round(x, 2)) + '%',
                          startangle=-270,
                          pctdistance=0.7,
                          labeldistance=1.1)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("filtered and called exons for the cluster " + clustID)

    pdf.savefig(fig)
    matplotlib.pyplot.close()
