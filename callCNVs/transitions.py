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
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
#   (NOCALL exons must have all likelihoods == -1)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - priors (ndarray dim nbStates): prior probabilities for each state
# - maxIED (int): max inter-exon distance for a pair of consecutive called exons
#   to contribute to the baseTransMatrix.
#
# Returns transMatrix (ndarray[floats] dim nbStates*nbStates): base transition
# probas between states
def buildBaseTransMatrix(likelihoods, exons, priors, maxIED):
    nbStates = len(priors)
    # count transitions between valid, close-enough exons in all samples
    countsAllSamples = numpy.zeros((nbStates, nbStates), dtype=numpy.uint64)

    for si in range(likelihoods.shape[0]):
        bestStates = (priors * likelihoods[si, :, :]).argmax(axis=1)
        prevChrom = ""
        prevEnd = 0
        prevState = 2
        for ei in range(len(exons)):
            # ignore NOCALL (ie all likelihoods == -1) exons
            if likelihoods[si, ei, 0] < 0:
                continue
            else:
                if exons[ei][0] != prevChrom:
                    # changed chrom
                    prevChrom = exons[ei][0]
                elif exons[ei][1] - prevEnd <= maxIED:
                    countsAllSamples[prevState, bestStates[ei]] += 1
                # in all cases, update prevs
                prevEnd = exons[ei][2]
                prevState = bestStates[ei]

    # Normalize each row to obtain the transition matrix
    baseTransMat = countsAllSamples.astype(numpy.float128)
    for i in range(nbStates):
        rowSum = countsAllSamples[i, :].sum()
        if rowSum > 0:
            baseTransMat[i, :] /= rowSum
        # else the row is all-zeroes, nothing to do
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
