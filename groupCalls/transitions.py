import logging
import numpy as np

import figures.plots

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
# getTransMatrix
# calculates the copy number transition matrix from a cluster and predictions for
# exons and samples based on likelihoods(probability density) and prior probabilities.
# It filters out irrelevant samples, determines the most probable copy number states
# for each exon, and normalizes the transition counts.
# The resulting transition matrix includes an additional 'void' state that incorporates
# the priors for all current observations, providing better initialization and
# resets during the Viterbi algorithm.
# Additionally, it generates a bar plot of copy number counts by samples.
#
# Args:
# - CNCallsArray (np.ndarray[floats]): likelihoods for each CN, each exon, each sample from a cluster
#                                      dim = [nbExons, (nbSamps * nbCNStates)]
# - priors (np.ndarray[floats]): prior probabilities for each copy number status.
# - sampsInClust (list[int]): cluster "samples" indexes
# - samples (list[str]): sample names
# - CNStatus (list[str]): Names of copy number types.
# - outFolder [str] : path to save the graphical representation.
#
# Returns:
# - normalized_arr (np.ndarray[floats]): transition matrix used for the hidden Markov model,
#                                        including the "void" state.
#                                        dim = [nbStates+1, nbStates+1]
def getTransMatrix(CNCallsArray, priors, sampsInClust, samples, CNStatus, outFile):
    nbSamps = len(sampsInClust)
    nbStates = len(CNStatus)
    nbExons = CNCallsArray.shape[0]

    # Initialize arrays
    sampCnCounts = np.zeros((nbSamps, nbStates), dtype=int)
    sampBestPath = np.full((nbExons, nbSamps), -1, dtype=np.int8)
    transitions = np.zeros((nbStates, nbStates), dtype=int)

    for sampIndexGp in range(len(sampsInClust)):
        sampleIndex = sampsInClust[sampIndexGp]
        sampProbsArray = CNCallsArray[:, sampleIndex * nbStates:sampleIndex * nbStates + nbStates]

        # filter no call samples
        if np.all(sampProbsArray == -1):
            print("Filtered samples:", samples[sampleIndex])
            continue

        # Filter -1 values
        exonCall = np.where(np.any(sampProbsArray != -1, axis=1))[0]

        # Calculate the weighted probabilities using filtered SampProbsArray and priors
        callProbsArray = sampProbsArray[exonCall, :]
        Odds = callProbsArray * priors

        maxCN = np.argmax(Odds, axis=1)

        # Updates
        unique_maxCN, counts = np.unique(maxCN, return_counts=True)
        sampCnCounts[sampIndexGp, unique_maxCN] += counts
        sampBestPath[exonCall, sampIndexGp] = maxCN
        prevCN = 2
        for indexExon in range(len(maxCN)):
            currCN = maxCN[indexExon]
            transitions[prevCN, currCN] += 1
            prevCN = currCN

    # Normalize each row to ensure sum equals 1
    row_sums = np.sum(transitions, axis=1, keepdims=True)
    normalized_arr = transitions / row_sums

    # Add void status and incorporate priors for all current observations
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
