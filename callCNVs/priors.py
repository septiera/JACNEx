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
# Refines initialization probabilities for a HMM using likelihood data.
# Methodology:
# - At the beginning of each iteration, the initialization probabilities are
# updated based on exon likelihood data and copy number type.
# - The new probabilities are compared to the probabilities from the previous
# iteration to determine if convergence has been achieved.
# - If the new probabilities match the previous ones (or if the difference is
# below a predefined threshold), convergence is assumed, and the probabilities are returned.
# - Otherwise, the process repeats with the new probabilities until convergence is
# reached or the maximum number of iterations is reached.
#
# Args:
# - likelihoods_A, likelihoods_G (dict[str:np.ndarray]): likelihood data for autosomal and gonosomal exons.
#                                                        Keys == sample IDs, values == numpy arrays of
#                                                        dim (nbExons * nbCNType).
# - jobs [int]: number of parallel jobs to run.
#
# Returns:
# - priors (numpy.ndarray[floats]): initialization probabilities for each CN type.
def getPriors(likelihoods_A, likelihoods_G, jobs):
    # Initialize the previous priors and format for convergence check
    priorsPrev = []
    formatPriorsPrev = ""
    maxIter = 20  # Maximum number of iterations allowed for convergence

    # Iterate through a maximum of 'maxIter' iterations
    for i in range(maxIter):
        # Update priors using likelihood data and previous priors
        priors = sumAndGetProbCNCounts(likelihoods_A, likelihoods_G, jobs, priorsPrev)
        # Format the priors for convergence check
        formatPriors = "\t".join(["%.2e" % x for x in priors])
        logger.info("Initialisation Matrix (Iteration n°{}):\n{}".format(i + 1, formatPriors))

        # Check for convergence by comparing formatted priors with the previous iteration
        if formatPriors == formatPriorsPrev:
            return priors

        # Update the previous formatted priors and priors list for the next iteration
        formatPriorsPrev = formatPriors
        priorsPrev = priors

    # If convergence is not achieved within the specified iterations, log a warning
    # and return the last computed priors (even if convergence was not achieved)
    logger.warning("No convergence after %d iterations", maxIter)
    return priors


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
####################
# sumAndGetProbCNCounts
# Methodology:
# - Counts the maximum likelihood CNs separately for autosomal and gonosomal exons.
# - Combines these counts to obtain the overall CN counts.
# - Normalizes these counts to derive the initialization probabilities.
# - Returns the computed priors.
#
# Args:
# - likelihoods_A, likelihoods_G (numpy.ndarray): Likelihood data
# - jobs [int]: number of parallel jobs to run.
# - priors_prev (numpy.ndarray[float]): prior probabilities from the previous iteration.
#
# Returns:
# - priors (numpy.ndarray[floats]): Initialization probabilities for the HMM.
def sumAndGetProbCNCounts(likelihoods_A, likelihoods_G, jobs, priorsPrev):
    # Count the maximum likelihoods for autosomal exons
    likelihoodsCounts_A = parallelizeCNCounts(likelihoods_A, priorsPrev, jobs)
    # Count the maximum likelihoods for gonosomal exons
    likelihoodsCounts_G = parallelizeCNCounts(likelihoods_G, priorsPrev, jobs)
    # Combine the counts from autosomal and gonosomal exons
    priorsCounts = likelihoodsCounts_A + likelihoodsCounts_G
    # Compute the probabilities by normalizing the counts (float128 not supported by numba)
    priors = priorsCounts.astype(numpy.float128) / priorsCounts.sum()
    return priors


############################################
# parallelizeCNCounts
# Perform parallelized counting of the most probable CN type across all samples.
# Methodology:
# - Distributes tasks across multiple jobs for efficiency.
# - Each sample's likelihood array is processed asynchronously.
# - Combines results to obtain counts of CNs with the highest likelihood.
#
# Args:
# - likelihoodDict (dict[str:np.ndarray]): keys == sampleID and values = numpy array
#                                          containing likelihood values for CNs and exons.
# - priorsPrev (numpy.ndarray[floats]): Prior probabilities from the previous iteration.
# - jobs [int]: number of parallel jobs to run.
#
# Returns:
# - count_array (numpy.ndarray[ints]): count of exons with the highest likelihood value
#                                      for each CN index across all samples.
def parallelizeCNCounts(likelihoodDict, priorsPrev, jobs):

    # Initialize an array to store the count of CNs with the highest likelihood value for each CN index
    maxCNIndex = numpy.zeros((likelihoodDict[list(likelihoodDict.keys())[0]].shape[1],), dtype=int)

    # determine the number of samples to process in parallel, based on available jobs
    paraSamp = min(math.ceil(jobs / 2), len(likelihoodDict.keys()))

    # Create a thread pool executor with the specified number of jobs
    with concurrent.futures.ThreadPoolExecutor(paraSamp) as pool:
        # Iterate over each numpy array in the dictionary
        futures = []
        for sampID, arr in likelihoodDict.items():
            # Submit a task to the pool to process each array asynchronously
            future = pool.submit(getCNCounts, arr, priorsPrev)
            futures.append(future)

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            # Retrieve the result of each completed task
            CN_counts = future.result()
            # Accumulate the counts
            maxCNIndex += CN_counts

    return maxCNIndex


#######################
# getCNCounts
# Process the likelihood array to count the occurrences of the most probable CN type.
#
# Args:
# - likelihoodArr (numpy.ndarray[floats]):containing likelihood values for each CNs types and exons.
# - priors (numpy.ndarray): Prior probabilities from the previous iteration.
#
# Returns:
# - CN_counts (numpy.ndarray[ints]): exon occurrences of the most probable CN type.
def getCNCounts(likelihoodArr, priorsPrev):
    CN_counts = numpy.zeros(likelihoodArr.shape[1], dtype=int)

    # Calculate the most probable CN path for the given likelihood array and priors.
    CNPath = getCNPath(likelihoodArr, priorsPrev)

    # Increment the count for each occurrence of the most probable CN type.
    for cn in CNPath:
        if cn == -1:
            continue
        CN_counts[cn] += 1

    return CN_counts


######################
# getCNPath
# Determines the most probable copy number type for each exon based on likelihoods and priors.
#
# Args:
# - likelihoodArr (numpy.ndarray[floats]):containing likelihood values for each CNs types and exons.
# - priorsPrev (numpy.ndarray[floats]): Prior probabilities from the previous iteration
#
# Returns:
# - CNPath (numpy.ndarray[ints]): containing the most probable CN type for each exon,
#                                 with -1 indicating non-interpretable data (nocall).
def getCNPath(likelihoodArr, priorsPrev):
    # Create a copy of the array to avoid modifying the original array
    arr_copy = numpy.copy(likelihoodArr)

    # Identify exons with non-interpretable data
    isSkipped = numpy.any(arr_copy == -1, axis=1)

    # Apply priors if available
    if len(priorsPrev) != 0:
        arr_copy *= priorsPrev

    # Determine the most probable CN type for each exon
    CNPath = numpy.argmax(arr_copy, axis=1)

    # WARN: Marks non-interpretable data with -1 but retains them in the list for the transition
    # calculation step to maintain the exon order for identifying chromosome changes.
    CNPath[isSkipped] = -1

    return CNPath
