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
#
#
# Also generates a bar plot of copy number counts by samples (only if logger debug mode).
#
# Args:
# - likelihoods(dict): keys= sampID, values= np.ndarray of likelihoods[floats]
#                         dim= nbOfExons * NbOfCNStates
# - exonOnChr

# - priors (np.ndarray[floats]): prior probabilities for each copy number status.
# - CNStates (list[strs]): Names of copy number types.
# - samp2clusts
# - outFile [str] : path to save the graphical representation.
#
# Returns:
# - transMatVoid (np.ndarray[floats]): transition matrix used for the hidden Markov model,
#                                      including the "void" state.
#                                      dim = [nbStates+1, nbStates+1]
def getTransMatrix(likelihoods_A, likelihoods_G, exonOnChr, priors, CNStates, samp2clusts, plotFile):
    nbStates = len(CNStates)
    # 2D array, expected format for a transition matrix [i; j]
    # contains all prediction counts of states, taking into account
    # the preceding states for the entire sampling.
    transitions = np.zeros((nbStates, nbStates), dtype=int)

    #
    countsCN_A, transitions = countsCNStates(likelihoods_A, nbStates, transitions, priors, exonOnChr)

    countsCN_G, transitions = countsCNStates(likelihoods_G, nbStates, transitions, priors, exonOnChr)

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
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        try:
            debug_process(transMatVoid, countsCN_A, countsCN_G, samp2clusts, CNStates, plotFile)
        except Exception as e:
            logger.error("DEBUG follow process failed: %s", repr(e))
            raise

    return transMatVoid


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
# countsCNStates
# Counts the Copy Number (CN) states for exons based on likelihood data and
# update transition matrix.

# Args:
# - likelihoodDict (dict): keys=sampID, values=np.ndarray(NBExons,NBStates)
# - nbStates [int]
# - transitions (np.ndarray[floats]): A transition matrix to be updated.
#                                     dim = NBStates * NBStates
# - priors (np.ndarray): Prior probabilities for CN states.
# - exonOnChr (dict): A dictionary mapping chromosome IDs to exon indexes.

# Returns:
# - countsCNDict (dict): A dictionary containing CN state counts for each sample.
# - transitions (np.ndarray): An updated transition matrix.
def countsCNStates(likelihoodDict, nbStates, transitions, priors, exonOnChr):
    # Initialize the dictionary for CN state counts
    # keys=sampID, values=np.array1D(NBStates)[ints]
    countsCNDict = {}

    for sampID in likelihoodDict.keys():
        # Initialize a vector for CN state counts
        countsVec = np.zeros(nbStates, dtype=int)

        # Iterate through chromosomes => resetting previous CN states for each new chromosome
        for chrID in exonOnChr:
            startChrExIndex = exonOnChr[chrID][0]
            endChrExIndex = exonOnChr[chrID][1]

            tmpArray = likelihoodDict[sampID][startChrExIndex:endChrExIndex, :]

            # Filter rows with -1 values, indicating filtered non-callable exons.
            exonCall = np.where(np.any(tmpArray != -1, axis=1))[0]

            if exonCall.shape[0] == 0:
                continue

            # Calculate the most probable CN state for each exon based on probabilities and priors.
            callProbsArray = tmpArray[exonCall, :]
            odds = callProbsArray * priors
            maxCN = np.argmax(odds, axis=1)

            # Update transition matrix and counts vector.
            # The initial previous state is set to CN2.
            prevCN = np.argmax(priors)
            for indexExon in range(len(maxCN)):
                currCN = maxCN[indexExon]
                transitions[prevCN, currCN] += 1
                prevCN = currCN
                countsVec[currCN] += 1

        # Store CN state counts for the sample.
        countsCNDict[sampID] = countsVec

    return (countsVec, transitions)


############################################
# debug_process
# Debugging function for tracking and visualizing data.
#
# Args:
# - transMatVoid (list of lists[floats]): transition matrix used for the hidden Markov model,
#                                         including the "void" state.
#                                         dim = [nbStates+1, nbStates+1]
# - countsCN_A (dict): CN state counts for autosomes.
# - countsCN_G (dict): CN state counts for gonosomes.
# - samp2clusts (dict): Sample-to-cluster mapping.
# - CNStates (list[strs]): List of CN states.
# - plotFile (str): File name for saving the plot.
#
# Returns:
# - None
def debug_process(transMatVoid, countsCN_A, countsCN_G, samp2clusts, CNStates, plotFile):
    # Log the transition matrix with formatting
    logger.debug("#### Transition Matrix (Formatted) #####")
    for row in transMatVoid:
        row2Print = ' '.join(format(num, ".3e") for num in row)
        logger.debug(row2Print)

    # Get all unique sample IDs from both dictionaries
    sampIDsList = set(countsCN_A.keys()).union(countsCN_G.keys())

    # Log the counting summary
    logger.debug("#### Counting Summary of CN Levels for Exons in Autosomes and Gonosomes #####")
    logger.debug(' '.join(["SAMPID", "clustID_A", "CN0_A", "CN1_A", "CN2_A", "CN3_A",
                           "clustID_G", "CN0_G", "CN1_G", "CN2_G", "CN3_G"]))

    countsList2Plot = []
    for sampID in sampIDsList:
        countsCN_A = countsCN_A.get(sampID, np.zeros(4, dtype=int))
        countsCN_G = countsCN_G.get(sampID, np.zeros(4, dtype=int))

        row2Print = (sampID + ' ' +
                     samp2clusts[sampID][0] + ' ' +
                     ' '.join("{:d}".format(CNcount) for CNcount in countsCN_A) + ' ' +
                     samp2clusts[sampID][1] + ' ' +
                     ' '.join("{:d}".format(CNcount) for CNcount in countsCN_G))
        logger.debug(row2Print)

        countsList2Plot.append(np.concatenate((countsCN_A, countsCN_G)).tolist())

    # Plot the data
    logger.debug("Plotting data...")
    try:
        figures.plots.barPlot(countsList2Plot, [term + "_A" for term in CNStates] + [term + "_G" for term in CNStates], plotFile)
    except Exception as e:
        logger.error("barPlot failed: %s", repr(e))
        raise
