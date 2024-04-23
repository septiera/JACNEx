import logging
import numpy
import math


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
########################
# adjustTransMatrix
# When distance between exons increases, the correlation between CN states of
# consecutive exons diminishes until at some point we simply expect the priors.
# Formally we implement this as follows:
#   - transition probabilities are smoothed from base probas to prior probas following
#   a power law until reaching the prior at d = dmax,
#
# Args:
# - transMatrix (np.array): base transition probabilities between states.
# - d [int]: distance between exons
# - dmax (float): Distance where prior is reached. Default 18000bp from Freeman GenomeRes 2006.
#
# Returns: adjusted transition probability matrix (np.ndarray dim(5*5))
# as defined above
def adjustTransMatrix(transMatrix, d, dmax=18000):
    # hardcoded power of the power law, values between 3-6 should be reasonable
    N = 5

    numStatesWithVoid = transMatrix.shape[0]
    newTrans = numpy.zeros_like(transMatrix)

    if d >= dmax:
        for prevState in range(numStatesWithVoid):
            newTrans[prevState, :] = transMatrix[0, :].copy()

    else:
        # newTrans[p->i](x) = trans[p->i] + (prior[i] - trans[p->i]) * (x/d_max)^N
        for state in range(1, numStatesWithVoid):
            newTrans[:, state] = (transMatrix[:, state] +
                                  (transMatrix[0, state] - transMatrix[:, state]) * (d / dmax) ** N)

        # squash first row with prior
        newTrans[0, :] = transMatrix[0, :].copy()

    return newTrans
