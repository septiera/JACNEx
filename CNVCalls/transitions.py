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
# - likelihoods(dict): keys= sampID, values= np.ndarray of likelihoods[floats]
#                         dim= nbOfExons * NbOfCNStates
# - priors (np.ndarray[floats]): prior probabilities for each copy number status.
# - CNStates (list[strs]): Names of copy number types.
# - outFile [str] : path to save the graphical representation.
#
# Returns:
# - transMatVoid (np.ndarray[floats]): transition matrix used for the hidden Markov model,
#                                      including the "void" state.
#                                      dim = [nbStates+1, nbStates+1]
def getTransMatrix(likelihoods_A, likelihoods_G, exonOnSexChr, exonOnChr, priors, CNStates, samp2clusts, plotFile):

    unique_keys_set = set(likelihoods_A.keys()).union(likelihoods_G.keys())
    unique_keys_list = list(unique_keys_set)
    nbSamps = len(unique_keys_list)
    nbStates = len(CNStates)

    # initialize arrays:
    # 2D array, each row represents a sample and each value is the count for a specific copy number type
    sampCnCounts = np.zeros((nbSamps, nbStates * 2), dtype=int)
    sampToIncrement = 0
    # 2D array, expected format for a transition matrix [i; j]
    # contains all prediction counts of states, taking into account
    # the preceding states for the entire sampling.
    transitions = np.zeros((nbStates, nbStates), dtype=int)

    for sampID in unique_keys_list:
        for chrID in exonOnChr:
            startChrExIndex = exonOnChr[chrID][0]
            endChrExIndex = exonOnChr[chrID][1]

            if exonOnSexChr[startChrExIndex] == 0:
                if sampID in likelihoods_A:
                    tmpArray = likelihoods_A[sampID][startChrExIndex:endChrExIndex, :]
                else:
                    continue
            else:
                if sampID in likelihoods_G:
                    tmpArray = likelihoods_G[sampID][startChrExIndex:endChrExIndex, :]
                else:
                    continue

            # filter row with -1 values, corresponding to filtered non-callable exons
            exonCall = np.where(np.any(tmpArray != -1, axis=1))[0]

            # calculate the most probable copy number state for each exon based on the
            # combined probabilities of the exon call likelihood and the prior probabilities
            callProbsArray = tmpArray[exonCall, :]
            odds = callProbsArray * priors
            maxCN = np.argmax(odds, axis=1)

            # updates arrays
            # The initial previous state is set to CN2.
            prevCN = np.argmax(priors)
            for indexExon in range(len(maxCN)):
                currCN = maxCN[indexExon]
                transitions[prevCN, currCN] += 1
                prevCN = currCN

                if chrID != "chrX" or chrID != "chrY":
                    sampCnCounts[sampToIncrement, currCN] += 1
                else:
                    sampCnCounts[sampToIncrement, currCN + 4] += 1

        row_str = (sampID + ' ' +
                   ' '.join(samp2clusts[sampID]) + ' ' +
                   ' '.join("{:d}".format(num) for num in sampCnCounts[sampToIncrement, :]))
        logger.debug(row_str)
        sampToIncrement += 1

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
            figures.plots.barPlot(sampCnCounts, CNStates, plotFile)
        except Exception as e:
            logger.error("barPlot failed: %s", repr(e))
            raise

    return transMatVoid
