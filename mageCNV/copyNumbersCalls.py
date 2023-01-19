import sys
import os
import logging
import numpy as np
import scipy.stats as st
from scipy.special import erf
import time

import mageCNV.slidingWindow

# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#####################################
# parseClustsFile ### !!! TO copy in clustering.py
# determines the type of clustering analysis performed in the previous step,
# gender discrimination or not.
# extraction of clustering informations
#
# Args:
# - clustsFile (str): a clusterFile produced by 2_clusterSamps.py
# - SOIs (List[str]): samples of interest names list
# Returns a tupple (clusts2Samps,clusts2Ctrls) , each variable is created here:
# - clusts2Samps (Dict[str, List[int]]): key: clusterID , value: samples index list
# - clusts2Ctrls (Dict[str, List[str]]): key: clusterID, value: controlsID list
# - nogender (boolean): identify if a discrimination of gender is made
def parseClustsFile(clustsFile, SOIs):
    try:
        clustsFH = open(clustsFile, "r")
        # Read the first line of the file, which contains the headers, and split it by tab character
        headers = clustsFH.readline().rstrip().split("\t")
    except Exception as e:
        # If there is an error opening the file, log the error and raise an exception
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception("cannot open clustsFile")

    # To Fill and returns
    # Initialize the dictionaries to store the clustering information
    clusts2Samps = {}
    clusts2Ctrls = {}

    # Initialize a flag to indicate whether gender is included in the clustering
    nogender = False

    # clusterFile shows that no gender discrimination is made
    if headers == ["samplesID", "clusterID", "controlledBy", "validitySamps"]:
        for line in clustsFH:
            # Split the line by tab character and assign the resulting
            # values to the following variables:
            # sampleID: the ID of the sample
            # validQC: sample valid status (0: invalid, 1: valid)
            # validClust: cluster valid status (0: invalid, 1: valid)
            # clusterID: clusterID ("" if not validated)
            # controlledBy: a list of cluster IDs that the current cluster is controlled by ("" if not validated)
            sampleID, validQC, validClust, clusterID, controlledBy = line.rstrip().split("\t", maxsplit=4)

            if validClust == "1":
                clusts2Samps.setdefault(clusterID, []).append(SOIs.index(sampleID))
                if controlledBy:
                    clusts2Ctrls[clusterID] = controlledBy.split(",")
            else:
                # If the sample is not valid, log a warning message
                logger.warning(f"{sampleID} dubious sample.")

    # clusterFile shows that gender discrimination is made
    elif headers == ["samplesID", "validQC", "validCluster_A", "clusterID_A",
                     "controlledBy_A", "genderPreds", "validCluster_G",
                     "clusterID_G", "controlledBy_G"]:
        nogender = True
        for line in clustsFH:
            # Split the line by tab character and assign the resulting
            # values to the following variables:
            # duplicate validClust, clusterID, controlledBy columns for autosomes (A) and gonosomes (G)
            # genderPreds: gender "M" or "F" ("" if not validated)
            split_line = line.rstrip("\n").split("\t", maxsplit=8)
            (sampleID, validQC, validClust_A, clusterID_A, controlledBy_A,
             genderPreds, validClust_G, clusterID_G, controlledBy_G) = split_line

            # populate clust2Samps
            # extract only valid samples for autosomes because it's
            # tricky to eliminate samples on their validity associated with gonosomes,
            # especially if one gender has few samples.
            if validClust_A == "1":
                # fill clusterID with fine description (A for autosomes,M for Male, F for Female)
                #################
                # Autosomes
                clusterID_A = f"{clusterID_A}A"
                clusts2Samps.setdefault(clusterID_A, []).append(SOIs.index(sampleID))
                if controlledBy_A:
                    clusts2Ctrls[clusterID_A] = [f"{ctrl}A" for ctrl in controlledBy_A.split(",")]

                #################
                # Gonosomes
                clusterID_G = f"{clusterID_G}{genderPreds}"
                clusts2Samps.setdefault(clusterID_G, []).append(SOIs.index(sampleID))
                if controlledBy_G:
                    clusts2Ctrls[clusterID_G] = [f"{ctrl}{genderPreds}" for ctrl in controlledBy_G.split(",")]
            else:
                logger.warning(f"{sampleID} dubious sample.")
    else:
        logger.error(f"Opening provided clustsFile {clustsFile}: {headers}")
        raise Exception("cannot open clustsFile")

    return(clusts2Samps, clusts2Ctrls, nogender)


#####################################
# CNCalls:
#
# Args:
#
#
#
# Returns
#
def CNCalls(counts_norm, clusterID, gono_index_flat, clusts2Samps, clusts2Ctrls, bandwidth, priors, emissionIssuesDict):
    ###############
    # select data corresponding to cluster
    # Select the rows of the normalized count matrix corresponding to the exons of the cluster
    if clusterID.endswith("A"):
        countsIncluster = np.delete(counts_norm, gono_index_flat, axis=0)
        exonsIndexes=np.delete(np.arange(0,len(counts_norm),1),gono_index_flat)
    else:
        countsIncluster = counts_norm[gono_index_flat]
        exonsIndexes=gono_index_flat
        
    # Get the indices of the samples in the cluster and its controls
    sampleIndexes = clusts2Samps[clusterID]
    if clusterID in clusts2Ctrls:
        for controls in clusts2Ctrls[clusterID]:
            sampleIndexes.extend(clusts2Samps[controls])
    sampleIndexes = list(set(sampleIndexes))

    # Select the columns of cluster_counts corresponding to the samples in the cluster and its controls
    countsIncluster = countsIncluster[:, sampleIndexes]

    ###########
    # Initialize InfoList with the exon index
    infoList = [[exon] for exon in exonsIndexes]

    ###########
    # fit a gamma distribution to find the profile of exons with little or no coverage (CN0)
    # - gammaParameters
    # - gammaThreshold
    gammaParameters, gammaThreshold = fitGammaDistributionPrivate(countsIncluster)

    ##############
    # Iterate over the exons
    for exon in range(len(exonsIndexes)):
        # Print progress every 10000 exons
        if exon % 10000 == 0:
            logger.info("%s: %s  %s ", clusterID, exon, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # Get count data for the exon
        exonFPM = countsIncluster[exon]

        ####################
        # Filter n°1: the exon is not covered => mu == 0
        # do nothing leave logOdds at zero
        mean_fpkm = exonFPM.mean()
        if mean_fpkm == 0:
            infoList[exon] += [0, 0, 0, 0, 0]
            continue

        ###################
        # Fit a robust Gaussian to the count data
        # - mean:
        # - stdev
        mean, stdev = fitRobustGaussianPrivate(exonFPM, bandwidth=bandwidth)

        ###################
        # Filter n°2: exon if standard deviation is zero
        # do nothing leave logOdds at zero
        if stdev == 0:
            # Define a new standard deviation to allow the calculation of the ratio
            if mean > gammaThreshold:
                stdev = mean / 20
            # Exon nocall
            else:
                infoList[exon] += [0, 0, 0, 0, 0]
                continue

        z_score = (mean - gammaThreshold) / stdev
        weight = computeWeightPrivate(exonFPM, mean, stdev)

        ###################
        # Filter n°3:
        # Exon nocall
        if (weight < 0.5) or (z_score < 3):
            infoList[exon] += [mean, stdev, z_score, weight]
            continue

        # Append values to InfoList
        infoList[exon] += [mean, stdev, z_score, weight]

        # Retrieve results for each sample
        for i in range(len(sampleIndexes)):
            sample_data = exonFPM[i]
            soi_name = sampleIndexes[i]

            log_odds = mageCNV.copyNumbersCalls.computeLogOddsPrivate(sample_data, gammaParameters, gammaThreshold, priors, mean, stdev)

            emissionIssuesDict[soi_name][exonsIndexes[exon]] = np.round(log_odds, 2)
    return(infoList)

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

############################
# fitGammaDistributionPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Estimate the parameters of the gamma distribution that best fits the data.
# The gamma distribution was chosen after testing 101 continuous distribution laws,
#  -it has few parameters (3 in total: shape, scale=1/beta, and loc),
#  -is well-known, and had the best goodness of fit on the empirical data.
# Arg:
# -countsIncluster (np.ndarray[floats]): cluster fragment counts (normalised)
# Returns a tupple (gamma_parameters, threshold_value), each variable is created here:
# -gammaParameters (tuple of floats): estimated parameters of the gamma distribution
# -thresholdValue (float): value corresponding to 95% of the cumulative distribution function
def fitGammaDistributionPrivate(countsIncluster):
    # compute meanFPM by exons
    # save computation time instead of taking the raw data (especially for clusters with many samples)
    meanCountByExons = np.mean(countsIncluster, axis=1)

    # smooth the coverage profile with kernel-density estimate using Gaussian kernels
    # - binEdges (np.ndarray[floats]): FPM range
    # - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
    #   dim= len(binEdges)
    binEdges, densityOnFPMRange = mageCNV.slidingWindow.smoothingCoverageProfile(meanCountByExons)

    # recover the threshold of the minimum density means before an increase
    # - minIndex (int): index from "densityMeans" associated with the first lowest
    # observed mean
    # - minMean (float): first lowest observed mean (not used for the calling step)
    (minIndex, minMean) = mageCNV.slidingWindow.findLocalMin(densityOnFPMRange)

    countsExonsNotCovered = meanCountByExons[meanCountByExons <= binEdges[minIndex]]

    countsExonsNotCovered.sort()  # sort data in-place

    # estimate the parameters of the gamma distribution that best fits the data
    gammaParameters = st.gamma.fit(countsExonsNotCovered)

    # compute the cumulative distribution function of the gamma distribution
    cdf = st.gamma.cdf(countsExonsNotCovered, a=gammaParameters[0], loc=gammaParameters[1], scale=gammaParameters[2])

    # find the index of the last element where cdf < 0.95
    thresholdIndex = np.where(cdf < 0.95)[0][-1]

    # compute the value corresponding to 95% of the cumulative distribution function
    # this value corresponds to the FPM value allowing to split covered exons from uncovered exons
    thresholdValue = countsExonsNotCovered[thresholdIndex]

    return (gammaParameters, thresholdValue)


############################
# fitRobustGaussianPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Fits a single principal gaussian component around a starting guess point
# in a 1-dimensional gaussian mixture of unknown components with EM algorithm
# script found to :https://github.com/hmiemad/robust_Gaussian_fit (v07.2022)
# Args:
#  -X (np.ndarray[float]): A sample of 1-dimensional mixture of gaussian random variables
#  -mu (float, optional): Expectation. Defaults to None.
#  -sigma (float, optional): Standard deviation. Defaults to None.
#  -bandwidth (float, optional): Hyperparameter of truncation. Defaults to 1.
#  -eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.
# Returns a tupple (mu,sigma), each variable is created here:
# mean [float] and stdev[float] of the gaussian component
def fitRobustGaussianPrivate(X, mu=None, sigma=None, bandwidth=1.0, eps=1.0e-5):
    # Integral of a normal distribution from -x to x
    weights = lambda x: erf(x / np.sqrt(2))
    # Standard deviation of a truncated normal distribution from -x to x
    sigmas = lambda x: np.sqrt(1 - 2 * x * st.norm.pdf(x) / weights(x))

    w, w0 = 0, 2

    if mu is None:
        # median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)

    if sigma is None:
        # rule of thumb
        sigma = np.std(X) / 3

    bandwidth_truncated_normal_weight = weights(bandwidth)
    bandwidth_truncated_normal_sigma = sigmas(bandwidth)

    while abs(w - w0) > eps:
        # loop until tolerence is reached
        try:
            """
            -create a window on X around mu of width 2*bandwidth*sigma
            -find the mean of that window to shift the window to most expected local value
            -measure the standard deviation of the window and divide by the standard
            deviation of a truncated gaussian distribution
            -measure the proportion of points inside the window, divide by the weight of
            a truncated gaussian distribution
            """
            W = np.where(np.logical_And(X - mu - bandwidth * sigma <= 0, X - mu + bandwidth * sigma >= 0), 1, 0)
            mu = np.mean(X[W == 1])
            sigma = np.std(X[W == 1]) / bandwidth_truncated_normal_sigma
            w0 = w
            w = np.mean(W) / bandwidth_truncated_normal_weight

        except:
            break

    return (mu, sigma)


############################
# computeWeightPrivate[PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# compute the sample contribution to the Gaussian obtained in a robust way.
#
# Args:
# - fpm_in_exon (np.ndarray[float]): FPM values for a particular exon for each sample
# - mean (float): mean FPM value for the exon
# - standard_deviation (float): std FPM value for the exon
# Returns weight of sample contribution to the gaussian for the exon [float]
def computeWeightPrivate(fpm_in_exon, mean, standard_deviation):
    targetData = fpm_in_exon[(fpm_in_exon > (mean - (2 * standard_deviation))) & 
                             (fpm_in_exon < (mean + (2 * standard_deviation))), ]
    weight = len(targetData) / len(fpm_in_exon)

    return weight


############################
# computeLogOddsPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Given four models, the log odds ratio (LOR) allows to choose the best-fitting model.
# Use of Bayes' theorem to deduce it.
#
# Args:
# - sample_data (float): a sample data point, FPM value
# - gamma_threshold (float):
# - prior_probabilities (list[float]): prior probabilities for different cases
# - mean (float): mean value for the normal distribution
# - standard_deviation (float): the standard deviation for the normal distribution
#
# Returns:
# - log_odds (list[float]): log-odds ratios for each copy number (CN0,CN1,CN2,CN3+)
def computeLogOddsPrivate(sample_data, params, gamma_threshold, prior_probabilities, mean, standard_deviation):
    # CN2 mean shift to get CN1 mean
    mean_cn1 = mean / 2

    # To Fill
    # Initialize an empty list to store the probability densities
    probability_densities = []

    ###############
    # Calculate the probability density for the gamma distribution (CN0 profil)
    # This is a special case because the gamma distribution has a heavy tail,
    # which means that the probability of density calculated from it can override
    # the other Gaussian distributions.
    # A condition is set up to directly associate a value of pdf to 0 if the sample FPM value
    # is higher than the mean of the Gaussian associated to CN1.
    # Reversely, the value of the pdf is truncated from the threshold value discriminating
    # covered from uncovered exons.
    gamma_pdf = 0
    cdf_cno_threshold = st.gamma.cdf(gamma_threshold, a=params[0], loc=params[1], scale=params[2])
    if sample_data <= mean_cn1:
        gamma_pdf = (1 / (1 - cdf_cno_threshold)) * st.gamma.pdf(sample_data, a=params[0], loc=params[1], scale=params[2])
    probability_densities.append(gamma_pdf)

    ################
    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    probability_densities.append(st.norm.pdf(sample_data, mean / 2, standard_deviation))
    probability_densities.append(st.norm.pdf(sample_data, mean, standard_deviation))
    probability_densities.append(st.norm.pdf(sample_data, 3 * mean / 2, standard_deviation))

    #################
    # Calculate the prior probabilities
    probability_densities_priors = []
    for i in range(len(probability_densities)):
        probability_densities_priors.append(probability_densities[i] * prior_probabilities[i])

    ##################
    # Calculate the log-odds ratios
    log_odds = []
    for i in range(len(probability_densities_priors)):
        # Calculate the denominator for the log-odds ratio
        to_subtract = probability_densities_priors[:i] + probability_densities_priors[i + 1:]
        to_subtract = np.sum(to_subtract)

        # Calculate the log-odds ratio for the current probability density
        if np.isclose(np.log10(to_subtract), 0):
            log_odd = 0
        else:
            log_odd = np.log10(probability_densities_priors[i]) - np.log10(to_subtract)

        log_odds.append(log_odd)

    return log_odds
