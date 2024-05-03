import concurrent.futures
import logging
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
# - likelihoodsDict: key==sampleID, value==(ndarray[floats] dim NbExons*NbStates)
#   holding the likelihoods of each state for each exon for this sample (no-call
#   exons should have all likelihoods == -1)
# - jobs (int): Number of jobs to run in parallel
#
# Returns priors (ndarray of NbStates floats): prior probabilities for each state.
def calcPriors(likelihoodsDict, jobs):
    # max number of iterations, hard-coded
    maxIter = 20

    # need nbStates, super ugly but it seems "this is the python way"
    nbStates = next(iter(likelihoodsDict.values())).shape[1]
    # priors start at 1 (ie we will initially count states with max likelihood)
    priors = numpy.ones(nbStates, dtype=numpy.float128)

    # to check for (approximate) convergence
    formattedPriorsPrev = ""
    # text including priors at each step, for logging in case we don't converge
    noConvergeString = ""
    # flag, true if we converged
    converged = False

    for i in range(maxIter):
        priors = calcPosteriors(likelihoodsDict, priors, jobs)
        formattedPriors = " ".join(["%.2e" % x for x in priors])
        debugString = "Priors at iteration " + str(i + 1) + ":\t" + formattedPriors
        logger.debug(debugString)
        noConvergeString += "\n" + debugString
        # Check for convergence
        if formattedPriors == formattedPriorsPrev:
            converged = True
            break
        else:
            formattedPriorsPrev = formattedPriors

    if not converged:
        logger.warning("Priors did not converge after %i iterations:%s", maxIter, noConvergeString)
        logger.warning("Try increasing maxIter in calcPriors(), and/or file an issue on github")
    # whether we converged or not, return last computed priors
    return priors


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################
# calcPosteriors:
# Given a likelihoodsDict and vector of prior probabilities, calculate and return
# the vector of posterior probabilities, which can be considered as an updated
# vector of priors. Samples are processed in parallel.
#
# Args:
# - likelihoodsDict: key==sampleID, value==(ndarray[floats] dim NbExons*NbStates)
#   holding the likelihoods of each state for each exon for this sample
# - priors (ndarray of NbStates floats): initial prior probabilities for each state
# - jobs (int): Number of jobs to run in parallel.
#
# Returns posteriors, same type as priors.
def calcPosteriors(likelihoodsDict, priors, jobs):
    countsPerState = numpy.zeros(len(priors), dtype=numpy.uint64)

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
        for sampID in likelihoodsDict.keys():
            futureRes = pool.submit(countMostLikelyStates, likelihoodsDict[sampID], priors)
            futureRes.add_done_callback(lambda f: sampleDone(f, countsPerState))

    posteriors = countsPerState.astype(numpy.float128) / countsPerState.sum()
    return(posteriors)


####################
# countMostLikelyStates:
# count for each CN state the number of exons whose most likely a posteriori state is CN.
#
# Args:
# - likelihoods (ndarray[floats] dim NbExons*NbStates): likelihoods of each state
#   for each exon for one sample
# - priors (ndarray of NbStates floats): initial prior probabilities for each state
#
# Returns the counts as an ndarray of NbStates uint64s
def countMostLikelyStates(likelihoods, priors):
    counts = numpy.zeros(len(priors), dtype=numpy.uint64)
    bestStates = (priors * likelihoods).argmax(axis=1)

    for ei in range(len(bestStates)):
        # ignore NOCALL (ie all likelihoods == -1) exons
        if likelihoods[ei][0] >= 0:
            counts[bestStates[ei]] += 1
    return(counts)
