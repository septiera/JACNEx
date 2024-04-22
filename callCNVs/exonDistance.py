import logging
import numpy
import math


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
########################
# initAdjustTransMatrix
# When distance between exons increases, the correlation between CN states of
# consecutive exons diminishes until at some point with simply expect the priors.
# Formally we implement this as follows:
# if transition probability to CN2 > priorCN2 leave untouched;
# else:
#   - transition to CN2 is inflated following an exponential law
#   until reaching the prior at d = dmax,
#   - transitions to CN0,CN1,CN3+ are reduced uniformly to compensate.
#
# Args:
# - transMatrix (np.array): Matrix of base transition probabilities between states.
# - dmax (float): Distance where prior is reached. Default 18000bp from Freeman GenomeRes 2006.
#
# Returns:
# - function: adjustTransMatrix(d)
#    - Arg: d [int]: distance between exons
#    - return: adjusted transition probability matrix (np.ndarray dim(5*5))
# as defined above
def initAdjustTransMatrix(transMatrix, dmax=18000):

    def adjustTransMatrix(d):
        priorCN2 = transMatrix[0, 3]
        numStatesWithVoid = transMatrix.shape[0]
        newTrans = numpy.zeros_like(transMatrix)

        if d >= dmax:
            for prevState in range(numStatesWithVoid):
                newTrans[prevState, :] = transMatrix[0, :].copy()

        else:
            # copy prior
            newTrans[0, :] = transMatrix[0, :].copy()

            for prevState in range(1, numStatesWithVoid):
                probTransCN2 = transMatrix[prevState, 3]
                if probTransCN2 > priorCN2:
                    newTrans[prevState, :] = transMatrix[prevState, :].copy()
                else:
                    fmax = priorCN2 / probTransCN2
                    f_d = fmax ** (d / dmax)
                    g_d = (1 - probTransCN2 * f_d) / (1 - probTransCN2)

                    newTrans[prevState, :] = g_d * transMatrix[prevState, :].copy()
                    newTrans[prevState, 3] = f_d * probTransCN2

        return newTrans

    return adjustTransMatrix
