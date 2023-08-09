import logging
import numpy as np

import figures.plots

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)
# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
# getTransMatrix
# Given likelihoods and priors, calculates the copy number transition matrix.
# Filters out irrelevant samples, determines most probable states for exons,
# and normalizes transitions.
# Includes a 'void' state incorporating priors for better initialization and restart
# in the personalised Viterbi algorithm.
# Also generates a bar plot of copy number counts by samples (only if logger debug mode).
#
# Args:
# - CNCallsArray (np.ndarray[floats]): likelihoods for each CN, each exon, each sample from a cluster
#                                      dim = [nbExons, (nbSamps * nbCNStates)]
# - priors (np.ndarray[floats]): prior probabilities for each copy number status.
# - samples (list[strs]): sample names
# - CNStatus (list[strs]): Names of copy number types.
# - outFile [str] : path to save the graphical representation.
#
# Returns:
# - transMatVoid (np.ndarray[floats]): transition matrix used for the hidden Markov model,
#                                      including the "void" state.
#                                      dim = [nbStates+1, nbStates+1]
def getTransMatrix(CNCallsArray, priors, samples, CNStatus, outFile):
    nbSamps = len(samples)
    nbStates = len(CNStatus)

    # initialize arrays:
    # 2D array, each row represents a sample and each value is the count for a specific copy number type
    sampCnCounts = np.zeros((nbSamps, nbStates), dtype=int)
    # 2D array, expected format for a transition matrix [i; j]
    # contains all prediction counts of states, taking into account
    # the preceding states for the entire sampling.
    transitions = np.zeros((nbStates, nbStates), dtype=int)

    for sampIndex in range(len(samples)):
        sampProbsArray = CNCallsArray[:, sampIndex * nbStates:sampIndex * nbStates + nbStates]

        # filter columns associated with no-call samples, same as those non-clusterable
        if np.all(sampProbsArray == -1):
            logger.debug("Filtered sample: %s", samples[sampIndex])
            continue

        # filter row with -1 values, corresponding to filtered non-callable exons
        exonCall = np.where(np.any(sampProbsArray != -1, axis=1))[0]

        # calculate the most probable copy number state for each exon based on the
        # combined probabilities of the exon call likelihood and the prior probabilities
        callProbsArray = sampProbsArray[exonCall, :]
        odds = callProbsArray * priors
        maxCN = np.argmax(odds, axis=1)

        # updates arrays
        # count of each predicted copy number type for the sample
        unique_maxCN, counts = np.unique(maxCN, return_counts=True)
        sampCnCounts[sampIndex, unique_maxCN] += counts
        # Incrementing the overall count of the sample's
        # The initial previous state is set to CN2.
        prevCN = np.argmax(priors)
        for indexExon in range(len(maxCN)):
            currCN = maxCN[indexExon]
            transitions[prevCN, currCN] += 1
            prevCN = currCN

    # normalize each row to ensure sum equals 1
    # not require normalization with the total number of samples
    row_sums = np.sum(transitions, axis=1, keepdims=True)
    normalized_arr = transitions / row_sums

    # add void status and incorporate priors for all current observations
    transMatVoid = np.vstack((priors, normalized_arr))
    transMatVoid = np.hstack((np.zeros((nbStates + 1, 1)), transMatVoid))

    #####################################
    ####### DEBUG PART ##################
    #####################################
    for row in transMatVoid:
        row_str = ' '.join(format(num, ".3e") for num in row)
        logger.debug(row_str)

    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        try:
            figures.plots.barPlot(sampCnCounts, CNStatus, outFile)
        except Exception as e:
            logger.error("barPlot failed: %s", repr(e))
            raise

    return transMatVoid
