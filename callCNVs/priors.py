import concurrent.futures
import logging
import numba
import numpy

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# calcPriors:
# calculate prior probabilities for each CN state.
# Algorithm starts by counting the states that maximize the likelihood, and
# then iteratively recalculates the priors by counting the states that maximize
# the posterior probability (ie prior * likelihood).
# This procedure ends when convergence is reached (approximately, ie using 2
# decimals in scentific notation for each proba) or after maxIter iterations.
#
# Args:
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
#   (NOCALL exons must have all likelihoods == -1)
# - jobs (int): number of jobs to run in parallel
#
# Returns priors (ndarray of nbStates floats): prior probabilities for each state.
def calcPriors(likelihoods, jobs):
    # max number of iterations, hard-coded
    maxIter = 20

    nbStates = likelihoods.shape[2]
    # priors start at 1 (ie we will initially count states with max likelihood)
    priors = numpy.ones(nbStates, dtype=numpy.float64)

    # to check for (approximate) convergence
    formattedPriorsPrev = ""
    # text including priors at each step, for logging in case we don't converge
    noConvergeString = ""
    # number of iterations to converge, 0 if we never converged
    converged = 0

    for i in range(maxIter):
        priors = calcPosteriors(likelihoods, priors, jobs)
        formattedPriors = " ".join(["%.2e" % x for x in priors])
        debugString = "Priors at iteration " + str(i + 1) + ":\t" + formattedPriors
        noConvergeString += "\n" + debugString
        # Check for convergence
        if formattedPriors == formattedPriorsPrev:
            converged = i + 1
            break
        else:
            formattedPriorsPrev = formattedPriors

    if converged:
        logger.debug("Priors converged after %i iterations:%s", converged, noConvergeString)
    else:
        logger.warning("Priors did not converge after %i iterations:%s", maxIter, noConvergeString)
        logger.warning("Try increasing maxIter in calcPriors(), and/or file an issue on github")
    # whether we converged or not, return last computed priors
    return priors


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################
# calcPosteriors:
# Given likelihoods and prior probabilities for each state, calculate the
# posterior probabilities for each state. These can then be considered as an
# updated vector of priors. Samples are processed in parallel.
#
# Args:
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
# - priors (ndarray of nbStates floats): initial prior probabilities for each state
# - jobs (int): number of jobs to run in parallel.
#
# Returns posteriors, same type as priors.
def calcPosteriors(likelihoods, priors, jobs):
    # init counts with a pseudo-count of one (avoid issues if no counts at all), this
    # won't matter for haploids since their CN1 likelihoods are zero
    countsPerState = numpy.ones(len(priors), dtype=numpy.uint64)

    ##################
    # Define nested callback for processing countMostLikelyStates() result.
    # sampleDone:
    # args: a Future object returned by ProcessPoolExecutor.submit(countMostLikelyStates),
    # and an ndarray that will be updated in-place (ie countsPerState).
    # If something went wrong, log and propagate exception, otherwise update countsPerState.
    def sampleDone(futureRes, counts):
        e = futureRes.exception()
        if e is not None:
            logger.error("countMostLikelyStates() failed for sample %s", str(e))
            raise(e)
        else:
            counts += futureRes.result()

    ##################
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        for si in range(likelihoods.shape[0]):
            futureRes = pool.submit(countMostLikelyStates, likelihoods[si, :, :], priors)
            futureRes.add_done_callback(lambda f: sampleDone(f, countsPerState))

    posteriors = countsPerState.astype(numpy.float64) / countsPerState.sum()
    return(posteriors)


####################
# countMostLikelyStates:
# count for each CN state the number of exons whose most likely a posteriori state is CN.
#
# Args:
# - likelihoods (ndarray[floats] dim nbExons*nbStates): likelihoods of each state
#   for each exon for one sample
# - priors (ndarray of nbStates floats): initial prior probabilities for each state
#
# Returns the counts as an ndarray of nbStates uint64s
@numba.njit
def countMostLikelyStates(likelihoods, priors):
    counts = numpy.zeros(len(priors), dtype=numpy.uint64)
    bestStates = (priors * likelihoods).argmax(axis=1)

    for ei in range(len(bestStates)):
        # ignore NOCALL (ie all likelihoods == -1) exons
        if likelihoods[ei, 0] >= 0:
            counts[bestStates[ei]] += 1
    return(counts)
