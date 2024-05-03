import concurrent.futures
import logging
import numpy

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# buildBaseTransMatrix
# Build the base transition matrix, using the states that maximize
# the posterior probability (ie prior * likelihood) for each
# pair of called exons that are "close enough" (maxIED).
#
# Args:
# - likelihoodsDict: key==sampleID, value==(ndarray[floats] dim NbExons*NbStates)
#   holding the likelihoods of each state for each exon for this sample (no-call
#   exons should have all likelihoods == -1)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - priors (ndarray dim nbStates): prior probabilities for each state
# - maxIED (int): max inter-exon distance for a pair of consecutive called exons
#   to contribute to the baseTransMatrix.
# - jobs (int): number of jobs to run in parallel.
#
# Returns transMatrix (ndarray[floats] dim nbStates*nbStates): base transition
# probas between states
def buildBaseTransMatrix(likelihoodsDict, exons, priors, maxIED, jobs):
    nbStates = len(priors)
    # count transitions between valid, close-enough exons in all samples
    countsAllSamples = numpy.zeros((nbStates, nbStates), dtype=numpy.uint64)

    ##################
    # Define nested callback for processing countMostLikelyTransitions() result.
    # sampleDone:
    # args: a Future object returned by ProcessPoolExecutor.submit(countMostLikelyTransitions),
    # and an ndarray that will be updated in-place (ie countsAllSamples).
    # If something went wrong, log and propagate exception.
    def sampleDone(futureRes, counts):
        e = futureRes.exception()
        if e is not None:
            logger.error("countMostLikelyTransitions() failed for sample %s", str(e))
            raise(e)
        else:
            counts += futureRes.result()

    ##################
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        for sampID in likelihoodsDict.keys():
            futureRes = pool.submit(countMostLikelyTransitions, likelihoodsDict[sampID],
                                    exons, priors, maxIED)
            futureRes.add_done_callback(lambda f: sampleDone(f, countsAllSamples))

    # Normalize each row to obtain the transition matrix
    baseTransMat = countsAllSamples.astype(numpy.float128) / countsAllSamples.sum(axis=1)
    return(baseTransMat)


######################################
# adjustTransMatrix
# When distance between exons increases, the correlation between CN states of
# consecutive exons diminishes until at some point we simply expect the priors.
# Formally we implement this as follows: transition probabilities are smoothed
# from base probas to prior probas following a power law, until reaching the
# prior at d = dmax.
#
# Args:
# - transMatrix (ndarray dim 4*4): base transition probabilities between states
# - priors (ndarray dim 4): prior probabilities for each state
# - d [int]: distance between exons
# - dmax (float): distance where prior is reached
#
# Returns: adjusted transition probability matrix (ndarray dim 4*4)
def adjustTransMatrix(transMatrix, priors, d, dmax):
    # hardcoded power of the power law, values between 3-6 should be reasonable
    N = 5

    newTrans = numpy.zeros_like(transMatrix)

    if d >= dmax:
        # use priors
        for prevState in range(len(transMatrix)):
            newTrans[prevState] = priors
    else:
        # newTrans[p->i](x) = trans[p->i] + (prior[i] - trans[p->i]) * (x/d_max)^N
        for state in range(len(transMatrix)):
            newTrans[:, state] = (transMatrix[:, state] +
                                  (priors[state] - transMatrix[:, state]) * (d / dmax) ** N)

    return newTrans


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

######################################
# countMostLikelyTransitions:
# Using the states that maximize the posterior probability (ie prior * likelihood),
# count the number of state transitions between valid, close-enough exons
#
# Args:
# - likelihoods (ndarray[floats] dim NbExons*NbStates): likelihoods of each state
#   for each exon for one sample
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - priors (ndarray dim nbStates): prior probabilities for each state
# - maxIED (int): max inter-exon distance for a transition to count
#
# Returns counts (ndarray[uint64] size nbStates*nbStates): numbers of (accepted)
# transitions between states
def countMostLikelyTransitions(likelihoods, exons, priors, maxIED):
    nbStates = len(priors)
    counts = numpy.zeros((nbStates, nbStates), dtype=numpy.uint64)
    bestStates = (priors * likelihoods).argmax(axis=1)

    prevChrom = ""
    prevEnd = 0
    prevState = 2
    for ei in range(len(exons)):
        # ignore NOCALL (ie all likelihoods == -1) exons
        if likelihoods[ei, 0] < 0:
            continue
        else:
            if exons[ei][0] != prevChrom:
                # changed chrom
                prevChrom = exons[ei][0]
            elif exons[ei][1] - prevEnd <= maxIED:
                counts[prevState, bestStates[ei]] += 1
            # in all cases, update prevs
            prevEnd = exons[ei][2]
            prevState = bestStates[ei]

    return(counts)
