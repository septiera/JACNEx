import logging
import numpy
import math
import concurrent.futures

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# getPriors
# refines initialization probabilities for a Hidden Markov Model (HMM) using likelihood data.
# It updates the priors based on exon likelihoods and copy number states in each iteration.
# The convergence check is based solely on whether the formatted priors
# (rounded to two decimal places in scientific notation) are equal to the formatted priors
# from the previous iteration.
# If they are equal, it is assumed that convergence has been achieved, and the function returns the priors.
# Otherwise, the iteration continues until the maximum number of iterations is reached.
# If convergence is not achieved within the specified number of iterations, a warning is logged.
#
# Args:
# - likelihoods_A, likelihoods_G (dict[str:np.ndarray]): keys==sampleID, values==numpy.ndarray(nbExons * nbCNStates)
# - CNStates (list[strs]): representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - jobs [int]: number of parallel jobs to run.
#
# Returns:
# - priors (numpy.ndarray): Initialization probabilities for the HMM.
def getPriors(likelihoods_A, likelihoods_G, CNStates, jobs):
    priorsPrev = []
    formatPriorsPrev = ""
    maxIter = 20

    for i in range(maxIter):
        priors = computePriors(likelihoods_A, likelihoods_G, CNStates, jobs, priorsPrev)
        formatPriors = "\t".join(["%.2e" % x for x in priors])
        logger.info("Initialisation Matrix (Priors n°{}):\n{}".format(i + 1, formatPriors))

        # Check for convergence
        if formatPriors == formatPriorsPrev:
            return priors

        formatPriorsPrev = formatPriors
        priorsPrev = priors

    logger.warning("No convergence after %d iterations", maxIter)
    return priors


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
####################
# computePriors
# It first counts the maximum likelihood copy numbers (CNs) for both autosomal
# and gonosomal exons separately.
# Then, it combines these counts to obtain the overall CN counts.
# Finally, it normalizes these counts to derive the initialization probabilities (priors) and returns them.
#
# Args:
# - likelihoods_A (numpy.ndarray): Likelihood data for autosomal exons.
# - likelihoods_G (numpy.ndarray): Likelihood data for gonosomal exons.
# - CNStates (list[str]): Represents the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - jobs [int]: number of parallel jobs to run.
# - priors_prev (list[float]): Prior probabilities from the previous iteration.

# Returns:
# - priors (numpy.ndarray): Initialization probabilities for the HMM.
def computePriors(likelihoods_A, likelihoods_G, CNStates, jobs, priorsPrev=[]):
    # Count the maximum likelihoods for autosomal exons
    likelihoodsCounts_A = countMaxLikelihoodCNParallel(likelihoods_A, CNStates, priorsPrev, jobs)
    # Count the maximum likelihoods for gonosomal exons
    likelihoodsCounts_G = countMaxLikelihoodCNParallel(likelihoods_G, CNStates, priorsPrev, jobs)
    # Combine the counts from autosomal and gonosomal exons
    priorsCounts = likelihoodsCounts_A + likelihoodsCounts_G
    # Compute the probabilities by normalizing the counts
    priors = priorsCounts.astype(numpy.float128) / priorsCounts.sum()
    return priors


############################################
# countMaxLikelihoodCNParallel
# Perform parallelized counting of the most probable copy number (CN) state
# for each CN index across all samples.
#
# Args:
# - sampLikelihoods_dict (dict): A dictionary where each key represents a sample identifier and
# the corresponding value is a numpy array containing likelihood values for CNs and exons.
# - CNStates (list[strs]): representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - priors (list[float]): Prior probabilities from the previous iteration.
# - jobs [int]: number of parallel jobs to run.
#
# Returns:
# - count_array (numpy.ndarray[ints]): A NumPy array containing the count of CNs with the highest
# likelihood value for each CN index across all samples.
def countMaxLikelihoodCNParallel(sampLikelihoods_dict, CNStates, priors, jobs):
    if len(sampLikelihoods_dict) == 0:
        return numpy.zeros(len(CNStates), dtype=int)

    # Initialize an array to store the count of CNs with the highest likelihood value for each CN index
    maxCNIndex = numpy.zeros((sampLikelihoods_dict[list(sampLikelihoods_dict.keys())[0]].shape[1],), dtype=int)

    # determine the number of clusters to process in parallel, based on available jobs
    paraSamp = min(math.ceil(jobs / 2), len(sampLikelihoods_dict.keys()))

    # Create a thread pool executor with the specified number of jobs
    with concurrent.futures.ThreadPoolExecutor(paraSamp) as pool:
        # Iterate over each numpy array in the dictionary
        futures = []
        for sampID, arr in sampLikelihoods_dict.items():
            # Submit a task to the executor to process each array asynchronously
            future = pool.submit(processSampLikelihoodArray, arr, CNStates, priors)
            futures.append(future)

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            # Retrieve the result of each completed task
            CN_counts = future.result()
            # Accumulate the counts
            maxCNIndex += CN_counts

    return maxCNIndex


#######################
# processSampLikelihoodArray
# Process the likelihood array to count the occurrences of the most probable copy number (CN)
# state for each CN index.
#
# Args:
# - arr (numpy.ndarray): A numpy array containing likelihood values for CNs and exons.
# - CNStates (list): A list representing the different copy number states (e.g., CN0, CN1, CN2, CN3+).
# - priors (numpy.ndarray): Prior probabilities for each CN state.

# Returns:
# - CN_counts (numpy.ndarray): An array containing the count of occurrences of the most probable CN
# state for each CN index.
def processSampLikelihoodArray(arr, CNStates, priors):
    # Create a copy of the array to avoid modifying the original array
    arr_copy = numpy.copy(arr)

    # Eliminate rows corresponding to no interpretable exons
    arr_copy = arr_copy[~numpy.any(arr_copy == -1, axis=1)]

    # Apply priors if available
    if len(priors) != 0:
        arr_copy *= priors

    # Determine the most probable CN state for each exon
    CNsList = numpy.argmax(arr_copy, axis=1)

    # Count the occurrences of each valid index
    CN_counts = numpy.zeros(len(CNStates), dtype=int)
    for cn in CNsList:
        CN_counts[cn] += 1

    return CN_counts
