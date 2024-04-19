import concurrent.futures
import logging
import math
import numpy
import scipy.stats
import traceback

####### JACNEx modules
import callCNVs.exonProfiling

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
##############################
# calcLikelihoods
# calculates the likelihoods of genomic segments (exons) in samples, considering
# both autosomes and gonosomes.
#
# Specs:
# - Processes autosomal and gonosomal data separately, allowing chromosome-specific analysis.
# - Employs FPM data
# - Utilizes a process pool for parallel computing, enhancing efficiency and reducing
#   processing time.
# - Individual clusters are processed.
# - Integrates parameters from statistical distributions to model CN profiles.
# - Computes likelihoods for each exon in each sample.
#
# Args:
# - samples (list[strs]): sample identifiers.
# - autosomeFPMs (numpy.ndarray[floats]): FPM data for autosomes. dim=[NbOfExons, NBOfSamples]
# - gonosomeFPMs (numpy.ndarray[floats]): same as autosomeFPMs but for gonosomes.
# - clust2samps (dict): key==clusterIDs, value==lists of sample IDs.
# - params_A (dict): key==clusterID, value== numpy.ndarray dim=[nbOfExons, NbOfParams]
#                     CN2 and CN0 rescale stdev in autosomes.
# - params_G (dict): same as params_G but for gonosomes.
# - CNStates (list[strs]): representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - jobs [int]: number of parallel jobs to run.
#
# Returns a tuple (likelihoods_A, likelihoods_G), where:
# - likelihoods_A (dict): key==sampleIDs, value==numpy.ndarray dim=[NBOfExons, NBOfCNStates]
#                         containing likelihoods for each sample in autosomes.
# - likelihoods_G (dict): same as likelihoods_A but for gonosomes.
def calcLikelihoods(samples, autosomeFPMs, gonosomeFPMs, clust2samps, clustIsValid, params_A, params_G, CNStates, jobs):
    # initialize output dictionaries
    likelihoods_A = {}
    likelihoods_G = {}

    # map sample names to indices for fast lookup
    sampIndexMap = callCNVs.exonProfiling.createSampleIndexMap(samples)

    # determine the number of clusters to process in parallel, based on available jobs
    paraClusters = min(math.ceil(jobs / 2), len(clust2samps))
    logger.info("%i new clusters => will process %i in parallel", len(clust2samps), paraClusters)

    # set up a pool of workers for parallel processing
    with concurrent.futures.ProcessPoolExecutor(paraClusters) as pool:
        # iterate over all clusters and submit them for processing
        for clusterID in clust2samps:

            # skip processing for invalid clusters
            if not clustIsValid[clusterID]:
                continue

            # determine the type of chromosome and process accordingly
            if clusterID.startswith("A"):
                # process autosomal clusters
                processClusterLikelihoods(clusterID, clust2samps, sampIndexMap, autosomeFPMs,
                                          params_A, len(CNStates), likelihoods_A, pool)
            else:
                # process gonosomal clusters
                processClusterLikelihoods(clusterID, clust2samps, sampIndexMap, gonosomeFPMs,
                                          params_G, len(CNStates), likelihoods_G, pool)

    return (likelihoods_A, likelihoods_G)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# processClusterLikelihoods
# Submits a task to calculate the likelihoods for a given cluster using parallel processing.
#
# Args:
# - clusterID (str): identifier of the cluster being processed.
# - clust2samps (dict): key==clusterIDs, value==lists of sample IDs.
# - samp2Index (dict): key==sampleID[str], value==exonsFPM samples column index[int].
# - exonsFPMs (numpy.ndarray): Normalized fragment counts for exons (FPM data).
# - hnorm_loc (float), hnorm_scale (float): Parameters for the half-Gaussian distribution.
# - params (dict): CN2 and CN0 rescale stdev, keyed by cluster ID.
# - numCNStates (int): The number of Copy Number (CN) states.
# - likelihoods (dict): Dictionary to store calculated likelihoods.
# - pool (ProcessPoolExecutor): Pool of workers for parallel processing.
# This function submits a calculation task to the pool and assigns a callback function
# to handle the results.
def processClusterLikelihoods(clusterID, clust2samps, samp2Index, exonsFPMs,
                              params, numCNStates, likelihoods, pool):
    future_res = pool.submit(calcClustLikelihoods, clusterID, samp2Index,
                             exonsFPMs, clust2samps, params[clusterID], numCNStates)
    future_res.add_done_callback(lambda future: updateLikelihoods(future, likelihoods, clusterID))


######################################
# updateLikelihoods
# Callback function to update the likelihoods dictionary with results from parallel processing.
#
# Args:
# - future_clusterLikelihoods (Future): A Future object containing the result or exception.
# - likelihoods (dict): Dictionary to store calculated likelihoods.
# - clusterID (str): identifier of the cluster.
# This function is called once the parallel processing task is completed. It checks for
# exceptions and updates the likelihoods dictionary with the calculated values.
def updateLikelihoods(future_clusterLikelihoods, likelihoods, clusterID):
    e = future_clusterLikelihoods.exception()
    if e is not None:
        logger.debug(traceback.format_exc())
        logger.warning("Analysis failed for cluster %s: %s.", clusterID, str(e))
    else:
        clusterLikelihoods = future_clusterLikelihoods.result()
        likelihoods.update(clusterLikelihoods)
        logger.info("Completed analysis for cluster %s.", clusterID)


######################################
# calcClustLikelihoods:
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
# - exonsFPM (numpy.ndarray[floats]): normalized fragment counts (FPMs)
# - clust2samps (dict): key==clusterID, value==list of sampleIDs
# - params (numpy.ndarray[floats]): Dim = nbOfExons * [stdevCN2, stdevCN0].
# - numCNs [int]: 4 copy number status ["CN0", "CN1", "CN2", "CN3"]
#
# Returns:
# - likelihoodsDict : keys==sampleID, values==numpy.ndarray(nbExons * nbCNStates)
#                     contains all the samples in a cluster.
def calcClustLikelihoods(clusterID, samp2Index, exonsFPM, clust2samps, params, numCNs):

    sampleIDs = clust2samps[clusterID]
    sampsIndexes = [samp2Index[samp] for samp in sampleIDs]

    # dictionary to hold the likelihoods for each sample, initialized to -1 for
    # all exons and CN states.
    likelihoodsDict = {samp: numpy.full((exonsFPM.shape[0], numCNs), -1, dtype=numpy.float128)
                       for samp in sampleIDs}

    # Identify exons with non-interpretable data
    isSkipped = numpy.any(params == -1, axis=1)

    # looping over each exon to compute the likelihoods for each CN state
    for ei in range(len(params)):
        if isSkipped[ei]:
            continue

        gauss_loc = 1
        gauss_scale = params[ei, 0]
        hnorm_loc = 0
        hnorm_scale = params[ei, 1]
        CN_params = setupCNDistribParams(hnorm_loc, hnorm_scale, gauss_loc, gauss_scale)

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

    return likelihoodsDict


######################################
# setupCNDistribParams
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
def setupCNDistribParams(hnorm_loc, hnorm_scale, gauss_loc, gauss_scale):

    # shifting Gaussian mean for CN1
    gaussShiftLoc = gauss_loc * 0.5

    # For the CN3+ distribution, a gamma distribution is used.
    # The gamma distribution is chosen for its ability to model data that are always positive
    # and might exhibit asymmetry, which is typical in certain data distributions.

    # The 'shape' parameter of the gamma distribution is set to 8.
    # This choice is the result of empirical testing, where different values were experimented
    # with, and 6 proved to provide the best fit to the data.
    # A higher 'shape' concentrates the distribution around the mean and reduces the spread,
    # which was found to be suitable for the characteristics of the CN3+ data.
    gamma_shape = 6

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
    gauss_logLocAdd1 = numpy.log10(gauss_locAddScale + 1)

    # Distribution parameters for CN0, CN1, CN2, and CN3+ are stored in CN_params.
    CN_params = [(scipy.stats.halfnorm.pdf, hnorm_loc, hnorm_scale, None),  # CN0
                 (scipy.stats.norm.pdf, gaussShiftLoc, gauss_scale, None),  # CN1
                 (scipy.stats.norm.pdf, gauss_loc, gauss_scale, None),  # CN2
                 (scipy.stats.gamma.pdf, gauss_locAddScale, gauss_logLocAdd1, gamma_shape)]  # CN3+
    return CN_params
