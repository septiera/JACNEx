import logging
import numpy as np
import scipy.stats
import concurrent.futures

####### with plots
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import os

import callCNVs.exonProcessing

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

##############################
# allChromLikelihoods
# process autosomes and gonosomes (sex chromosomes).
# It computes likelihoods of various copy number states for each genomic segment by
# considering both the autosomal and gonosomal FPM data.
# The function integrates multiple analytical steps, including mapping sample IDs,
# processing autosomes and gonosomes separately, and returning the computed likelihoods for each.
#
# Args:
# - samples (list[strs]): sample identifiers.
# - autosomeFPMs (np.ndarray[floats]): FPM data for autosomes. dim=[NbOfExons, NBOfSamples]
# - gonosomeFPMs (np.ndarray[floats]): same as autosomeFPMs but for gonosomes.
# - clust2samps (dict): key==clusterIDs, value==lists of sample IDs.
# - fitWith (dict): key==clusterIDs, value= list of cluster IDs to fit with the current cluster.
# - hnorm_loc, hnorm_scale [float][float]: Parameters for the half-normal distribution.
# - CN2Params_A (dict): key==clusterID, value== np.ndarray dim=[nbOfExons, NbOfParams]
#                     Parameters for CN2 state (representing normal copy number) in autosomes.
# - CN2Params_G (dict): same as CN2Params_A but for gonosomes.
# - CNStates (list[strs]): representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - priors (list[floats]): List of prior probabilities for each copy number state.
# - jobs [int]: number of parallel jobs to run.
# - plotDir [str]: Directory path where plots (if any) will be saved.
#
# Returns a tuple (likelihoods_A, cnPaths_A, likelihoods_G, cnPaths_G), where:
# - likelihoods_A (dict): key==sampleIDs, value==np.ndarray dim=[NBOfExons, NBOfCNStates]
#                         containing likelihoods for each sample in autosomes.
# - cnPaths_A (dict): key==sampleIDs, value==value==1D NumPy arrays representing the most
#                     probable CN state path for each exon on autosomes.
# - likelihoods_G (dict): same as likelihoods_A but for gonosomes.
# - cnPaths_G (dict): same as cnPaths_A but for gonosomes.
def allChromLikelihoods(samples, autosomeFPMs, gonosomeFPMs, clust2samps, fitWith, hnorm_loc, hnorm_scale,
                        CN2Params_A, CN2Params_G, CNStates, priors, jobs, plotDir):

    # Mapping sample IDs to their indexes.
    samp2Index = {samples[i]: i for i in range(len(samples))}

    # Process autosomes
    likelihoods_A, cnPaths_A = analyzeWeightCNData(samp2Index, autosomeFPMs, clust2samps, fitWith, hnorm_loc,
                                                   hnorm_scale, CN2Params_A, CNStates, priors, jobs, plotDir)

    # Process gonosomes
    likelihoods_G, cnPaths_G = analyzeWeightCNData(samp2Index, gonosomeFPMs, clust2samps, fitWith, hnorm_loc,
                                                   hnorm_scale, CN2Params_G, CNStates, priors, jobs, plotDir)

    return (likelihoods_A, cnPaths_A, likelihoods_G, cnPaths_G)


#############################
# analyzeWeightCNData
# It calculates likelihoods for different copy number states, determines the best
# copy number paths, counts copy number occurrences, and computes sample weights 
# of belonging to the cluster. Finally, it weights the likelihoods with these weights,
# providing a refined understanding of the exon copy number in the dataset,
# taking into account the confidence level of a call.
#
# Args:
# - samples (list[strs]): sample identifiers.
# - fpmData(np.ndarray[floats]): dim=[NbOfExons, NBOfSamples]
# - clust2samps(dict): key==clusterIDs, value==lists of sample IDs.
# - fitWith (dict): key==clusterIDs, value= list of cluster IDs to fit with the current cluster.
# - hnorm_loc, hnorm_scale[float][float]: Parameters for the half-normal distribution.
# - CN2Params(dict): key==clusterID, value== np.ndarray dim=[nbOfExons, NbOfParams]
#                    Parameters for CN2 state (representing normal copy number).
# - CNStates (list[strs]): representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - priors (list[floats]): List of prior probabilities for each copy number state.
# - jobs [int]: number of parallel jobs to run.
# - plotDir [str]: Directory path where plots (if any) will be saved.
#
# Returns:
# - weightedLikelihoods (dict): key==sampleIDs, value==np.ndarray dim=[NBOfExons, NBOfCNStates]
#                               containing likelihoods for each sample.
# - weightedCnPaths (dict):key==sampleIDs, value==value==1D NumPy arrays representing the most
#                           probable CN state path for each exon.
#                           Exons with no calls are represented by -1.
def analyzeWeightCNData(samples, fpmData, clust2samps, fitWith, hnorm_loc, hnorm_scale, CN2Params,
                        CNStates, priors, jobs, plotDir):
    # compute likelihoods in parallel
    try:
        likelihoods = computeLikelihoodsParallel(clust2samps, samples, fpmData, hnorm_loc,
                                                 hnorm_scale, CN2Params, len(CNStates), jobs)
    except Exception as e:
        logger.error("computeLikelihoodsParallel failed:%s", repr(e))

    # determine the best CN paths and count CN occurrences per sample
    try:
        cnPaths = determineSampCNPaths(likelihoods, priors)
        cnCounts = calcSampCNOccurrences(cnPaths, len(CNStates))
    except Exception as e:
        logger.error("determineSampCNPaths or calcSampCNOccurrences failed:%s", repr(e))

    # Debug: Logging initial CN counts
    printCNCounts(cnCounts, clust2samps)

    # analyze cluster CN statistics, and calculate sample probabilities
    try:
        sampWeights = computeClustCNWeights(list(CN2Params.keys()), clust2samps, fitWith, cnCounts, plotDir)
    except Exception as e:
        logger.error("computeClustCNStatsProbs failed:%s", repr(e))

    # Weight likelihoods with the product of probabilities
    try:
        weightedLikelihoods = applyProbWeights2Likelihoods(likelihoods, sampWeights)
    except Exception as e:
        logger.error("applyProbWeights2Likelihoods failed:%s", repr(e))

    # Debug: Recalculate CN paths and occurrences with weighted likelihoods
    try:
        weightedCnPaths = determineSampCNPaths(weightedLikelihoods, priors)
        weightedCnCounts = calcSampCNOccurrences(weightedCnPaths, len(CNStates))
    except Exception as e:
        logger.error("determineSampCNPaths or calcSampCNOccurrences failed:%s", repr(e))

    # Logging recalculated CN counts after weighting
    printCNCounts(weightedCnCounts, clust2samps)

    # Debug: Find indexes where value == 0 different between cnPaths et weightedCnPaths
    sampTarget = "grexome0116"
    diff_indices = np.where((cnPaths[sampTarget] == 0) != (weightedCnPaths[sampTarget] == 0))[0]
    logger.debug(f'diff homodels after weighting likelihoods for {sampTarget}: {diff_indices}')
    prevHetDelIndex = np.where(cnPaths[sampTarget] == 1)[0]
    logger.debug(f'total hetdel before weighting likelihoods for {sampTarget}: {prevHetDelIndex}')
    diff_indices = np.where((cnPaths[sampTarget] == 1) != (weightedCnPaths[sampTarget] == 1))[0]
    logger.debug(f'diff hetdel after weighting likelihoods for {sampTarget}: {diff_indices}')
    prevDupIndex = np.where(cnPaths[sampTarget] == 3)[0]
    logger.debug(f'total dup before weighting likelihoods for {sampTarget}: {prevDupIndex}')
    diff_indices = np.where((cnPaths[sampTarget] == 3) != (weightedCnPaths[sampTarget] == 3))[0]
    logger.debug(f'diff dup after weighting likelihoods for {sampTarget}: {diff_indices}')

    return (weightedLikelihoods, weightedCnPaths)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

######################################
# computeLikelihoodsParallel
# Parallelizes the computation of likelihoods across clusters using multiprocessing.
#
# Args:
# - clust2samps (dict): key==clusterIDs, value=sampleIDs list.
# - samp2Index (dict): key==sampleIDs, value==exonsFPMs sample column index.
# - exonsFPMs (np.ndarray): FPM data for exons.
# - exonMetrics (dict): key==clsuetrIDs, value==paramaters of Gaussian distribution.
# - numCNStates [int]: CN states number.
# - jobs [int]: Number of jobs to run in parallel.
#
# Returns:
# - likelihoodsDict (dict): key==sampleIDs[str], value==likelihoods arrays, dim=NBOfExon*NBOfCNState.
#                           contains all the samples present in exonsFPMs.
def computeLikelihoodsParallel(clust2samps, samp2Index, exonsFPMs, hnorm_loc, hnorm_scale,
                               exonMetrics, numCNStates, jobs):
    # Initialize dictionary to store likelihoods.
    likelihoodsDict = {}

    # Determine the number of clusters to process in parallel.
    paraClusters = min(max(jobs // 2, 1), len(clust2samps))  # Ensure at least one cluster is processed
    logger.info("%i clusters => will process %i in parallel", len(clust2samps), paraClusters)

    # Define function to merge results from parallel computation into likelihoods dictionary.
    def updateLikelihoodsDict(futureResult):
        try:
            clusterID, clusterLikelihoods = futureResult.result()
            likelihoodsDict.update(clusterLikelihoods)
            logger.info("Done calcClustCNLikelihoods %s", clusterID)
        except Exception as e:
            logger.warning("Failed calcClustCNLikelihoods: %s", repr(e))

    # Start parallel computation of likelihoods.
    with concurrent.futures.ProcessPoolExecutor(max_workers=paraClusters) as executor:
        futures = []
        for clusterID, clustCN2Params in exonMetrics.items():
            future = executor.submit(calcClustCNLikelihoods, clusterID, samp2Index,
                                     exonsFPMs, clust2samps, hnorm_loc, hnorm_scale,
                                     clustCN2Params, numCNStates)
            future.add_done_callback(updateLikelihoodsDict)
            futures.append(future)

        # Wait for all futures to complete.
        concurrent.futures.wait(futures)
    return likelihoodsDict


######################################
# calcClustCNLikelihoods:
# calculates the likelihoods (probability density values) for various copy number
# scenarios for each exon's FPM within a sample, using different statistical distributions.
# The probability density functions (PDFs) are computed as follows for each copy number
# (CN) state:
#   - CN0: Uses a half-normal distribution, typically fitted to intergenic regions where
#          no genes are present, indicating the absence of the genomic segment.
#   - CN2: Utilizes a Gaussian distribution that is robustly fitted to represent the typical
#          coverage observed in the genomic data (loc = mean, scale = standard deviation).
#   - CN1: Employs the Gaussian parameters from CN2 but adjusts the mean (loc) by a factor of 0.5,
#          representing a single copy loss.
#   - CN3+: For scenarios where copy numbers exceed 2, it applies a gamma distribution
#           with parameters derived from the CN2 Gaussian distribution to model the heavy-tailed
#           nature of higher copy number events.
#           This involves:
#           - An alpha parameter (shape in SciPy), which determines the tail behavior of the distribution.
#             Alpha > 1 indicates a heavy tail, alpha = 1 resembles a Gaussian distribution,
#             and alpha < 1 suggests a light tail.
#           - A theta parameter (scale in SciPy) that modifies the spread of the distribution.
#             A higher theta value expands the distribution, while a lower value compresses it.
#           - A 'loc' parameter in SciPy that shifts the distribution along the x-axis,
#             modifying the central tendency without altering the shape or spread.
#
# Args:
# - clusterID [str]: cluster identifier being processed
# - samp2Index (dict): key==sampleID[str], value==exonsFPM samples column index[int]
# - exonsFPM (np.ndarray[floats]): normalized fragment counts (FPMs)
# - clust2samps (dict): key==clusterID, value==list of sampleIDs
# - hnorm_loc[float], hnorm_scale[float]: parameters of the half normal distribution
# - clustCN2Params (np.ndarray[floats]): Dim = nbOfExons * ["loc", "scale"].
#                                   Gaussian distribution parameters.
# - numCNs [int]: 4 copy number status ["CN0", "CN1", "CN2", "CN3"]
#
# Returns a tupple (clusterID, likelihoodClustDict):
# - clusterID [str]
# - likelihoodsDict : keys==sampleID, values==np.ndarray(nbExons * nbCNStates)
#                     contains all the samples in a cluster.
def calcClustCNLikelihoods(clusterID, samp2Index, exonsFPM, clust2samps, hnorm_loc, hnorm_scale,
                           clustCN2Params, numCNs):
    sampleIDs = clust2samps[clusterID]
    sampsIndexes = [samp2Index[samp] for samp in sampleIDs]

    # dictionary to hold the likelihoods for each sample, initialized to -1 for
    # all exons and CN states.
    likelihoodsDict = {samp: np.full((exonsFPM.shape[0], numCNs), -1, dtype=np.float128)
                       for samp in sampleIDs}

    # check for valid exons, which do not contain -1, indicating no call exons.
    validExons = ~np.any(clustCN2Params == -1, axis=1)

    # looping over each exon to compute the likelihoods for each CN state
    for ei in range(len(clustCN2Params)):
        if not validExons[ei]:
            continue

        gauss_loc = clustCN2Params[ei, 0]
        gauss_scale = clustCN2Params[ei, 1]
        CN_params = setupCNDistFunctions(hnorm_loc, hnorm_scale, gauss_loc, gauss_scale)

        for ci, (pdfFunction, loc, scale, shape) in enumerate(CN_params):
            exonFPMs = exonsFPM[ei, sampsIndexes]
            if shape is not None:
                # Apply gamma distribution
                likelihoods = pdfFunction(exonFPMs, loc=loc, scale=scale, a=shape)
            else:
                # Apply normal or half-normal distribution
                likelihoods = pdfFunction(exonFPMs, loc=loc, scale=scale)

            for si, likelihood in enumerate(likelihoods):
                sampID = sampleIDs[si]
                likelihoodsDict[sampID][ei, ci] = likelihood

    return (clusterID, likelihoodsDict)


######################################
# setupCNDistFunctions
# Defines parameters for four types of distributions (CN0, CN1, CN2, CN3+),
# involving half normal, normal, and gamma distributions.
# For CN3+, the parameters are empriricaly adjusted to ensure compatibility with
# Gaussian distribution.
#
# Args:
# - hnorm_loc[float], hnorm_scale[float]: parameters of the half normal distribution
# - gauss_loc [float]: Mean parameter for the Gaussian distribution (CN2).
# - gauss_scale [float]: Standard deviation parameter for the Gaussian distribution (CN2).
# Returns:
# - CN_params(list of list): contains distribution objects from Scipy representing
#                            different copy number types (CN0, CN1, CN2, CN3+).
#                            Parameters vary based on distribution type (ordering: loc, scale, shape).
def setupCNDistFunctions(hnorm_loc, hnorm_scale, gauss_loc, gauss_scale):

    # shifting Gaussian mean for CN1
    gaussShiftLoc = gauss_loc * 0.5

    # For the CN3+ distribution, a gamma distribution is used.
    # The gamma distribution is chosen for its ability to model data that are always positive
    # and might exhibit asymmetry, which is typical in certain data distributions.

    # The 'shape' parameter of the gamma distribution is set to 8.
    # This choice is the result of empirical testing, where different values were experimented
    # with, and 8 proved to provide the best fit to the data.
    # A higher 'shape' concentrates the distribution around the mean and reduces the spread,
    # which was found to be suitable for the characteristics of the CN3+ data.
    gamma_shape = 8

    # The 'loc' parameter is defined as the sum of 'gauss_loc' and 'gauss_scale'.
    # This sum shifts the gamma distribution to avoid significant overlap with the Gaussian
    # CN2 distribution, allowing for a clearer distinction between CN2 and CN3+.
    gauss_locAddScale = gauss_loc + gauss_scale

    # The 'scale' parameter is determined by the base 10 logarithm of 'gauss_locAddScale + 1'.
    # Adding '+1' is crucial to prevent issues with the logarithm of a zero or negative value.
    # The use of the logarithm helps to reduce the scale of values, making the model more
    # adaptable and stable, especially if 'gauss_locAddScale' varies over a wide range.
    # This approach creates a narrower distribution more in line with the expected
    # characteristics of CN3+ data.
    gauss_logLocAdd1 = np.log10(gauss_locAddScale + 1)

    # Distribution parameters for CN0, CN1, CN2, and CN3+ are stored in CN_params.
    CN_params = [(scipy.stats.halfnorm.pdf, hnorm_loc, hnorm_scale, None),  # CN0
                 (scipy.stats.norm.pdf, gaussShiftLoc, gauss_scale, None),  # CN1
                 (scipy.stats.norm.pdf, gauss_loc, gauss_scale, None),  # CN2
                 (scipy.stats.gamma.pdf, gauss_locAddScale, gauss_logLocAdd1, gamma_shape)]  # CN3+
    return CN_params


#######################
# determineSampCNPaths
# Computes the most probable copy number state (CN state) paths for each sample
# based on provided likelihoods and priors.
#
# Args:
# - likelihoodDict (dict): key==sampleIDs, value==np.array2D of likelihoods.
#                          dim = nbExons * nbCNStates
# - priors (list[floats]): Prior probabilities for each CN states.
#
# Returns:weightLikelihoodsWithProbabilities(likelihoods, sampProbToCN)
# - sampCNPathDict: key==sampleIDs, value==1D NumPy arrays representing the most
#                   probable CN state path for each exon.
#                   Exons with no calls are represented by -1.
def determineSampCNPaths(likelihoodDict, priors):
    sampCNPathDict = {}

    # Iterate through each sample (sampID) and its likelihoods
    for sampID, likelihoods in likelihoodDict.items():
        # Multiply likelihoods by priors. Priors are broadcasted to match likelihoods' shape
        odds = likelihoods * priors

        # Identify no-call positions (-1) and reset them in the odds array
        no_calls = likelihoods == -1

        # Find the index of the CN state with the highest probability for each exon
        CNsList = np.argmax(odds, axis=1)
        CNsList[no_calls[:, 0]] = -1  # Reset no calls to -1

        sampCNPathDict[sampID] = CNsList

    return sampCNPathDict


#######################
# calcSampCNOccurrences
# Counts the occurrences of each copy number state for each sample based on
# the CN state paths.
#
# Args:
# - sampCNPathDict (dict): key==sampleIDs, value== numpy array1D (nbExons) of CNs[ints]
#
# Returns:
# - sampCNCountsDict (dict): key==sampleIDs, value==numpy array1D with the counts of
#                            occurrences for each CN state.
#                            The index in the counts array corresponds to the CN state.
def calcSampCNOccurrences(sampCNPathDict, numCNs):
    sampCNCountsDict = {}
    for sampID, CNpath in sampCNPathDict.items():
        # Use np.bincount to count occurrences of each value, excluding -1
        filtered_arr = CNpath[CNpath != -1]
        counts = np.bincount(filtered_arr, minlength=numCNs)  # Ensure all CN states are counted

        sampCNCountsDict[sampID] = counts

    return sampCNCountsDict


##############################################
# computeClustCNWeights
# Processes data for a single cluster: calculates CN statistics
# and computes weights corresponding of sample membership in the cluster.
# + DEBUG code: generates histogram plots.
#
# Args:
# - clusters (list[strs]): clusterIDs list
# - clust2samps (dict): key==clusterIDs, value==lists of sampleIDs.
# - cnCounts (dict): key==sampleIDs, value==CN counts for each sample.
# - plotDir (str): Directory to save the plot files.
#
# Returns:
# - sampProbToCN (dict): key==sampleIDs, value==probabilities for CN states.
def computeClustCNWeights(clusters, clust2samps, fitWith, cnCounts, plotDir):
    sampCNWeights = {}
    for clusterID in clusters:
        # efficiently retrieve CN counts for the current cluster
        sampleIDsCurrClust = clust2samps[clusterID]
        currClusterCounts = np.array([cnCounts[sid] for sid in sampleIDsCurrClust])

        # Combine CN counts from additional clusters specified in 'fitWith'
        fitWithClusterCountsList = []
        for fitWithClustID in fitWith[clusterID]:
            for sid in clust2samps[fitWithClustID]:
                if sid in cnCounts:
                    fitWithClusterCountsList.append(cnCounts[sid])
        fitWithClusterCounts = np.array(fitWithClusterCountsList)

        # Combine counts only if additional clusters are valid
        if len(fitWithClusterCounts) > 0:
            combinedCounts = np.vstack((currClusterCounts, fitWithClusterCounts))
        else:
            combinedCounts = currClusterCounts

        # Compute Gaussian statistics for each CN state
        mean = np.ones(currClusterCounts.shape[1])
        stdev = np.ones(currClusterCounts.shape[1])
        for i in range(combinedCounts.shape[1]):
            (gaussian_loc, gaussian_scale) = callCNVs.exonProcessing.fitRobustGaussian(combinedCounts[:, i])
            mean[i] = gaussian_loc
            stdev[i] = gaussian_scale
        logger.debug(f"##### clusterID {clusterID}: mean {' '.join(map(str, mean))}, stdev {' '.join(map(str, stdev))}")

        # Compute probabilities for each sample in the current cluster
        for sid in sampleIDsCurrClust:
            sampCNWeights[sid] = computeSampCNZscore(cnCounts[sid], mean, stdev)
            logger.debug(f"{sid} {' '.join(map(str, sampCNWeights[sid]))}")

        # DEBUG: Generate histograms for the current cluster
        if logger.isEnabledFor(logging.DEBUG):
            try:
                genClustCNHistograms(clusterID, currClusterCounts, fitWithClusterCounts,
                                     mean, stdev, plotDir)
            except Exception as e:
                logger.error(f"Failed generating histograms and stats for cluster {clusterID}: {e}")

    return sampCNWeights


################################
# computeSampCNZscore
# calculate weights for each CN state of a sample based on z-scores.
# This function penalizes samples that significantly deviate from the expected Gaussian distribution
# (i.e., those with high z-scores) and leaves the weights for low z-scores (close to the mean) unchanged.
#
# Args:
# - cnCounts (np.ndarray[ints]):  counts for each CN state.
# - mean (np.ndarray[floats]): mean values for each CN state, used in the Gaussian distribution.
# - stdev (np.ndarray[floats]): standard deviation values for each CN state, used in the Gaussian distribution.
#
# Returns:
# - sampProbToCN (np.ndarray[floats]): Gaussian sampling probabilities for CN states belonging
#                                      to the cluster.
def computeSampCNZscore(cnCounts, mean, stdev):
    # Define a threshold for the z-score
    # zscore_threshold = 2
    weights = np.ones(len(cnCounts), dtype=np.float128)

    for i in range(len(cnCounts)):
        if i != 2:  # Exclude CN2 from the calculation
            # Calculate the z-score for the current CN state
            zscore = abs(cnCounts[i] - mean[i]) / stdev[i]

            # # Penalize high z-scores, which indicate significant deviation from the Gaussian distribution
            # if zscore > zscore_threshold:
            #     # Apply a penalty for high z-scores
            #     weights[i] = 1 / (1 + zscore - zscore_threshold)
            weights[i] = 1 / (1 + zscore)

    return weights


###########################
# applyProbWeights2Likelihoods
# weights the likelihoods for each sample with the product of probabilities
# for belonging to CN0, CN1, and CN3
#
# Args:
# - likelihoods (dict): key==sampleIDs, values==2D NumPy arrays of likelihoods
#                       dim=[numberOfExons, numberOfCNStates].
# - sampsWeights (dict): key==sampleIDs, values== np.ndarray 1D of weigth value.
#
# Returns:
# - weightedLikelihoods (dict): key==sampleIDs, values==2D NumPy arrays of likelihoods weighted
#                               dim=[numberOfExons, numberOfCNStates].
def applyProbWeights2Likelihoods(likelihoods, sampsWeights):
    weightedLikelihoods = {}

    for sid, sLikelihoods in likelihoods.items():
        # Skip weighting if -1 is present in the likelihoods (mask those rows)
        mask = np.all(sLikelihoods != -1, axis=1)

        # Create a copy of sLikelihoods
        sampsLikelihoods = sLikelihoods.copy()

        # Apply the weighting only to the rows without -1
        sampsLikelihoods[mask] *= sampsWeights[sid]

        weightedLikelihoods[sid] = sampsLikelihoods

    return weightedLikelihoods


###############################################################################
############################## DEBUG FUNCTIONS ################################
###############################################################################

#######################
# print CN counts in log
def printCNCounts(dict, clust2samps):
    for sampleID, countArray in dict.items():
        # Find all cluster IDs for the current sample
        clusterIDs = [clustID for clustID, samples in clust2samps.items() if sampleID in samples]

        # Convert the NumPy array to a Python list for readability
        countList = countArray.tolist()
        # Convert the list to a string with spaces between elements and no brackets
        countStr = ' '.join(map(str, countList))
        # Log each sample's CN count along with its cluster IDs
        logger.info("Sample ID: %s, CN Counts: %s, Clusters: %s", sampleID, countStr, ' '.join(clusterIDs))


#######################
# genClustCNHistograms DEBUG function
# Plot histograms and statistics for a cluster's copy number counts.
# Take into account the clusters that are fit with by generating a second histogram,
# plot the Gaussian distribution tailored to the cluster and CN, and generate a
# histogram for each CN (0, 1, 3), saving a PDF for each cluster.
#
# Args:
# - clusterID [str]
# - currClusterCounts (numpy.ndarray): 2D array containing copy number counts
#                                      for each sample in a cluster
# - fitWithClusterCounts (numpy.ndarray): 2D array containing copy number counts
#                                         for each sample in fitWith clusters.
#                                         can be empty si no fitWith clusters.
# - mean (numpy.ndarray): mean values for each CNs.
# - upper_threshold (numpy.ndarray): upper threshold values for each CNs.
# - cluster [str]
# - backendPDF (matplotlib.backends.backend_pdf.PdfPages): PDF backend for saving plots.
def genClustCNHistograms(clusterID, currClusterCounts, fitWithClusterCounts, mean, stdev, plotDir):
    histogramsFile = os.path.join(plotDir, f"CN_counts_{clusterID}.pdf")
    with matplotlib.backends.backend_pdf.PdfPages(histogramsFile) as backendPDF:
        # Iterate over CNs (0, 1, 3)
        for ci in range(currClusterCounts.shape[1]):
            if ci == 2:
                continue  # Skip CN2

            matplotlib.pyplot.figure(figsize=(8, 6))

            # Extract column counts for the current CN
            currColumnCounts = currClusterCounts[:, ci]

            # Ensure fitWithClusterCounts is not empty and has the same dimensions
            if fitWithClusterCounts.shape[0] > 0:
                fitWithColumnCounts = fitWithClusterCounts[:, ci]
                all_data = np.hstack([fitWithColumnCounts, currColumnCounts])
            else:
                all_data = currColumnCounts

            # Determine histogram parameters
            bin_range = (np.min(all_data), np.max(all_data))

            # Calculate the number of bins
            max_size = max(currClusterCounts.shape[0], fitWithClusterCounts.shape[0])
            min_bins = max(10, int(max_size / 2))

            # Define bins
            bins = np.linspace(bin_range[0], bin_range[1], min_bins)

            # Plot histograms
            if fitWithClusterCounts.shape[0] > 0:
                matplotlib.pyplot.hist(fitWithColumnCounts, bins=bins, alpha=0.5, color='g',
                                       label='Fit Counts', density=True)

            if len(currColumnCounts) > 1:
                matplotlib.pyplot.hist(currColumnCounts, bins=bins, alpha=0.5, color='b',
                                       label='Current Counts', density=True)
            else:
                matplotlib.pyplot.axvline(currColumnCounts, color='b', linestyle='dashed',
                                          linewidth=2, label='Single Count')

            # Calculate the maximum value for the Gaussian fit
            max_fit_with = np.max(fitWithColumnCounts) if len(fitWithClusterCounts) > 1 else 0
            max_curr = np.max(currColumnCounts)
            data_max = max(max_fit_with, max_curr)
            xi = np.linspace(0, data_max, 1000)

            # Plot Gaussian fit
            pdfGaussian = scipy.stats.norm.pdf(xi, loc=mean[ci], scale=stdev[ci])
            matplotlib.pyplot.axvline(mean[ci], color='r', linestyle='dashed', linewidth=2, label='Mean')
            matplotlib.pyplot.plot(xi, pdfGaussian, label=f"Gaussian (loc={mean[ci]:.2f}, scale={stdev[ci]:.2f})")

            # Set plot labels and legends
            matplotlib.pyplot.title(f'Cluster {clusterID} - {len(currColumnCounts)} samples - CN{ci} Histogram')
            matplotlib.pyplot.xlabel('Exon Counts')
            matplotlib.pyplot.ylabel('Density')
            matplotlib.pyplot.legend()

            # Save the plot to PDF and close the figure
            backendPDF.savefig()
            matplotlib.pyplot.close()
