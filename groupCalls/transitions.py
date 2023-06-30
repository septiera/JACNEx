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
#############################
# getTransitionMatrix
# Given a matrix that contains log-likelihood values for each copy number status
# per exon per sample, along with prior information on the copy numbers,
# calculates the observed data based on copy number (CN) values and generates
# a transition matrix.
# It performs the following steps:
# - converting input arrays into observed CN values,
# - counting occurrences of each CN type,
# - merging transition matrices,
# - and optionally generating a bar plot.
# It provides insights into the observed CN data and the transitions between CN types.
#
# Args :
# - CNCallsArray (np.ndarray[floats]): log-likelihoods for each CN, each exon, each sample
#                                     dim = [nbExons, (nbSamps * nbCNStates)]
# - priors (np.ndarray[floats]): log-Prior probabilities for each copy number status.
# - samples (list[str]): sample names
# - CNStatus (list[str]): Names of copy number types.
# - outFolder [str] : path to save the graphical representation.
#
# Returns a tuple (obsDataBased, transtionMatrix):
# - obsDataBased (np.ndarray[int]) : observation lists for each sample (conserved no call as -1).
# - transitionMatrix (np.ndarray[floats])
def getTransitionMatrix(CNCallsArray, priors, samples, CNStatus, outFolder):
    try:
        # Convert the callsArrays into observed CN values based on priors and samples.
        # Returns an np.ndarray of observation lists (columns) in CN (integers).
        # The dimensions are NBexons x NBsamples.
        obsDataBased = probs2CN(CNCallsArray, samples, priors)
    except Exception as e:
        logger.error("probs2CN failed: %s", repr(e))
        raise

    try:
        # Count the occurrences of each CN type for each sample.
        # Returns an np.ndarray of integers representing the count of each CN type for each sample.
        # The dimensions are NBsamples x NBCNStates.
        # Also returns a list of np.ndarray representing the transitions between CN types for each sample.
        (countArray, transitionMatrices) = countCNObservations(obsDataBased, len(CNStatus))
    except Exception as e:
        logger.error("countCNObservations failed: %s", repr(e))
        raise

    try:
        # Merge the individual transition matrices into a single transition matrix.
        # Returns a np.ndarray representing the merged transition matrix.
        transitionMatrix = mergeTransMatrices(transitionMatrices)
    except Exception as e:
        logger.error("mergeTransMatrices failed: %s", repr(e))
        raise

    if (logging.getLogger().getEffectiveLevel() <= logging.DEBUG):
        try:
            # Generate a bar plot of the countArray.
            figures.plots.barPlot(countArray, CNStatus, outFolder)
        except Exception as e:
            logger.error("barPlot failed: %s", repr(e))
            raise

    return (obsDataBased, transitionMatrix)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################
# probs2CN
# Compute the observations of copy number (CN) states for each exon in each sample,
# considering both forward and backward directions of exon reading.
#
# Args:
# - CNCallsArray (ndarray[floats]): containing the probabilities (log-likelihoods) of CN states for
#                                   each exon and sample. dim = [NBExons, NBSamples * NBStates]
# - samples (list[str]): sample names
# - priors (np.ndarray[floats]): shape (nbStates,) containing the weights (log) for calculating
#                                odds.
#
# Returns:
# - CN_list (np.ndarray[ints]): observed CN states for each exon and sample.
#                                    The observed CN state is represented as an index.
#                                    dim = [NBExons, NBSamples]
def probs2CN(CNCallsArray, samples, priors):
    NBSamples = len(samples)
    NbExons = CNCallsArray.shape[0]
    NbStates = len(priors)

    # List to store CN observations for each sample
    CN_list = np.full((NbExons, NBSamples), -1, dtype=np.int8)

    # Set to store filtered sample indices
    filtered_samples = set()

    for sampleIndex in range(NBSamples):
        exonCallInd = 0
        SampProbsArray = CNCallsArray[:, sampleIndex * NbStates:sampleIndex * NbStates + NbStates]
        # Check if the entire probsArray is -1 for each sample
        if np.all(SampProbsArray == -1):
            filtered_samples.add(sampleIndex)
            # Print the names of filtered samples
            print("Filtered samples:", samples[sampleIndex])
            continue
        
        # Filter -1 values
        exonCallInd =  np.where(np.any(SampProbsArray != -1, axis=1))[0]
        
        # Calculate the weighted probabilities using filtered probsArray and priors
        callprobsArray = SampProbsArray[exonCallInd, :]
        logOdds = callprobsArray + priors
        
        CNpred = np.argmax(logOdds, axis=1)
        CN_list[exonCallInd, sampleIndex] = CNpred

    return CN_list


#############################
# countCNObservations
# Count the observations of copy number (CN) states for each patient and CN type,
# and calculate the transition matrix for each sample.
#
# Args:
# - CN_list (np.ndarray[int]): Array of shape (nbExons, nbSamples) containing the observed CN states
#     for each exon and sample.
# - nbStates [int]: Number of CN states.
#
# Returns a tuple (countArray, transitionMatrices):
# - countArray (np.ndarray[ints]): the count of each CN type for each sample, dim = NBsamps x NBCNStates
# - transitionMatrices (list of np.ndarray[ints]): transition matrices, each of shape (nbStates, nbStates),
#                                                  representing the transitions between CN types for each sample.
def countCNObservations(CN_list, nbStates):
    nbSamples = CN_list.shape[1]

    # Array to store the count of each CN type for each sample
    countArray = np.zeros((nbSamples, nbStates), dtype=int)

    # List to store the transition matrices
    transitionMatrices = []

    for sampleIndex in range(nbSamples):
        CNObserved4Samp = CN_list[:, sampleIndex]
        CNcalls =  CNObserved4Samp[CNObserved4Samp != -1]
        counts = np.bincount(CNcalls, minlength= nbStates)
        countArray[sampleIndex] = counts

        # Calculate transition matrix
        countMatrix = np.zeros((nbStates, nbStates), dtype=int)

        prevCN = 2
        for currCN in CNcalls:
            countMatrix[prevCN, currCN] += 1
            prevCN = currCN

        transitionMatrices.append(countMatrix)
        
    return transitionMatrices


######################################################
# mergeTransMatrices
# Merges a list of transition matrices by taking the sum of each element in the list.
# The resulting matrix is then normalized, ensuring that the sum of each row equals 1.
# Then it's log transformed
#
# Args:
# - transMatricesList (list[np.ndarray[int]]): List of transition matrices to be merged.
#
# Returns:
# - normalized_arr (np.ndarray[floats]): Merged and normalized transition matrix.
def mergeTransMatrices(transMatricesList):
    merged_array = np.sum(transMatricesList, axis=0)

    # Normalize each row to ensure sum equals 1
    row_sums = np.sum(merged_array, axis=1, keepdims=True)
    normalized_arr = merged_array / row_sums

    # log Transformation
    log_array = np.log(normalized_arr)

    return (log_array)
