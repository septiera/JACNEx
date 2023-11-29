import logging
import numpy as np
import scipy.stats
import concurrent.futures

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

##############################
# allChrom
# process autosomes and gonosomes (sex chromosomes).
# It computes likelihoods of various copy number states for each genomic segment by
# considering both the autosomal and gonosomal FPM data.
#
# Args:
# - samples (list[strs]): sample identifiers.
# - autosomeFPMs (np.ndarray[floats]): FPM data for autosomes. dim=[NbOfExons, NBOfSamples]
# - gonosomeFPMs (np.ndarray[floats]): same as autosomeFPMs but for gonosomes.
# - clust2samps (dict): key==clusterIDs, value==lists of sample IDs.
# - hnorm_loc, hnorm_scale [float][float]: Parameters for the half-normal distribution.
# - CN2Params_A (dict): key==clusterID, value== np.ndarray dim=[nbOfExons, NbOfParams]
#                     Parameters for CN2 state (representing normal copy number) in autosomes.
# - CN2Params_G (dict): same as CN2Params_A but for gonosomes.
# - CNStates (list[strs]): representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - jobs [int]: number of parallel jobs to run.
#
# Returns a tuple (likelihoods_A, likelihoods_G), where:
# - likelihoods_A (dict): key==sampleIDs, value==np.ndarray dim=[NBOfExons, NBOfCNStates]
#                         containing likelihoods for each sample in autosomes.
# - likelihoods_G (dict): same as likelihoods_A but for gonosomes.
def allChrom(samples, autosomeFPMs, gonosomeFPMs, clust2samps, hnorm_loc, hnorm_scale, CN2Params_A,
             CN2Params_G, CNStates, jobs):

    # Process autosomes
    likelihoods_A = calcLikelihoodsInParallel(clust2samps, samples, autosomeFPMs, hnorm_loc, hnorm_scale,
                                              CN2Params_A, len(CNStates), jobs)

    # Process gonosomes
    likelihoods_G = calcLikelihoodsInParallel(clust2samps, samples, gonosomeFPMs, hnorm_loc, hnorm_scale,
                                              CN2Params_G, len(CNStates), jobs)

    return (likelihoods_A, likelihoods_G)


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
        CN_params = setupCNDistribFunctions(hnorm_loc, hnorm_scale, gauss_loc, gauss_scale)

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


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

######################################
# calcLikelihoodsInParallel
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
def calcLikelihoodsInParallel(clust2samps, samp2Index, exonsFPMs, hnorm_loc, hnorm_scale, exonMetrics,
                              numCNStates, jobs):
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
# setupCNDistribFunctions
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
def setupCNDistribFunctions(hnorm_loc, hnorm_scale, gauss_loc, gauss_scale):

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
