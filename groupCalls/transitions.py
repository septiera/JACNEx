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
# Calculates the copy number transition matrix and copy number predictions for
# exons and samples.
# It iterates over each sample, filters out samples with all probabilities set to -1,
# and calculates the weighted probabilities.
# It determines the most probable copy number state for each exon and updates the
# copy number counts by samples and copy number predictions for exons and samples.
# Finally, it normalizes the copy number transition counts, calculates the logarithm
# (base 10) of the normalized array for the transition matrix.
# With  logger debug mode, prints the normalized copy number transition counts
# and the transition matrix used for the hidden Markov model (HMM).
# It generates a bar plot of the copy number counts by samples.
#
# Args:
# - CNCallsArray (np.ndarray[floats]): log-likelihoods for each CN, each exon, each sample
#                                     dim = [nbExons, (nbSamps * nbCNStates)]
# - priors (np.ndarray[floats]): log-Prior probabilities for each copy number status.
# - samples (list[str]): sample names
# - CNStatus (list[str]): Names of copy number types.
# - outFolder [str] : path to save the graphical representation.
#
# Returns a tuple (transMatrix, exons2CN4Samps)
# - transMatrix (np.ndarray[floats]): transition matrix used for the hidden Markov model.
#                                     dim = [nbStates, nbStates]
# - exons2CN4Samps(np.ndarray[int]) : observation lists for each sample (conserved no call as -1).
#                                     dim = [NbExons, NbSamps]
def getTransMatrix(CNCallsArray, priors, samples, CNStatus, outFolder):
    NBSamps = len(samples)
    NBStates = len(CNStatus)
    NBExons = CNCallsArray.shape[0]

    # Initialize arrays
    CNCountsBySamps = np.zeros((NBSamps, NBStates), dtype=int)
    exons2CN4Samps = np.full((NBExons, NBSamps), -1, dtype=np.int8)  # to return
    countTransCN = np.zeros((NBStates, NBStates), dtype=int)

    filtered_samples = set()  # Set to store filtered sample indexes

    for sampleIndex in range(NBSamps):
        exonCallInd = 0
        SampProbsArray = CNCallsArray[:, sampleIndex * NBStates:sampleIndex * NBStates + NBStates]

        # Check if the entire probsArray is -1 for each sample
        if np.all(SampProbsArray == -1):
            filtered_samples.add(sampleIndex)
            print("Filtered samples:", samples[sampleIndex])
            continue

        # Filter -1 values
        exonCallInd = np.where(np.any(SampProbsArray != -1, axis=1))[0]

        # Calculate the weighted probabilities using filtered SampProbsArray and priors
        callprobsArray = SampProbsArray[exonCallInd, :]
        logOdds = callprobsArray + priors

        CNpred = np.argmax(logOdds, axis=1)

        CNCountsBySamps[sampleIndex, :] = np.bincount(CNpred, minlength=NBStates)
        exons2CN4Samps[exonCallInd, sampleIndex] = CNpred

        prevCN = 2
        for currCN in CNpred:
            countTransCN[prevCN, currCN] += 1
            prevCN = currCN

    # Normalize each row to ensure sum equals 1
    row_sums = np.sum(countTransCN, axis=1, keepdims=True)
    normalized_arr = countTransCN / row_sums
    # Calculate the logarithm (base 10) of the normalized array for HMM transition matrix
    transMatrix = np.log10(normalized_arr)  # to return

    #####################################
    ####### DEBUG PART ##################
    #####################################
    logger.debug("Normalized Copy Number Transition Counts for All Samples:")
    for row in normalized_arr:
        row_str = ' '.join(format(num, ".3e") for num in row)
        logger.debug(row_str)

    logger.debug("Transition matrix used for HMM:")
    for row in transMatrix:
        row_str = ' '.join(format(num, ".3e") for num in row)
        logger.debug(row_str)

    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        try:
            figures.plots.barPlot(CNCountsBySamps, CNStatus, outFolder)
        except Exception as e:
            logger.error("barPlot failed: %s", repr(e))
            raise

    return (transMatrix)
