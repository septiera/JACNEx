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
# Given a matrix that contains likelihood values for each copy number status
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
# - callsArrays (np.ndarray[floats]): likelihoods for each CN, each exon, each sample
#                                     dim = [nbExons, (nbSamps * nbCNStates)]
# - priors (np.ndarray[floats]): Prior probabilities for each copy number status.
# - samples (list[str]): sample names
# - CNStatus (list[str]): Names of copy number types.
# - outFolder [str] : path to save the graphical representation.
#
# Returns a tuple (obsDataBased, transtionMatrix):
# - obsDataBased (np.ndarray[int]) : observation lists for each sample (conserved no call as -1).
# - transtionMatrix (np.ndarray[floats])
def getTransitionMatrix(callsArrays, priors, samples, CNStatus, outFolder):
    try: 
        # Convert the callsArrays into observed CN values based on priors and samples.
        # Returns an np.ndarray of observation lists (columns) in CN (integers).
        # The dimensions are NBexons x NBsamples.
        obsDataBased = probs2CN(callsArrays, samples, priors, len(CNStatus))
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
        transtionMatrix = mergeTransMatrices(transitionMatrices)
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

    return (obsDataBased, transtionMatrix)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################
# probs2CN
# Compute the observations of copy number (CN) states for each exon in each sample,
# considering both forward and backward directions of exon reading.
#
# Args:
# - CNcalls (ndarray[floats]): containing the probabilities (likelihoods) of CN states for
#                              each exon and sample. dim = [NBExons, NBSamples * NBStates]
# - samples (list[str]): sample names
# - priors (np.ndarray[floats]): shape (nbStates,) containing the weights for calculating
#                                odds.
# - nbStates [int]: Number of CN states.
#
# Returns:
# - observations (np.ndarray[ints]): observed CN states for each exon and sample.
#                                    The observed CN state is represented as an index.
#                                    dim = [NBExons, NBSamples]
def probs2CN(CNcalls, samples, priors, nbStates):
    nbSamples = len(samples)

    # List to store CN observations for each sample
    observations_list = []

    # Set to store filtered sample indices
    filtered_samples = set()

    # Check if the entire probsArray is -1 for each sample
    for sampleIndex in range(nbSamples):
        probsArray = CNcalls[:, sampleIndex * nbStates:sampleIndex * nbStates + nbStates]
        if np.all(probsArray == -1):
            filtered_samples.add(sampleIndex)
            # Print the names of filtered samples
            print("Filtered samples:", samples[sampleIndex])
        else:
            # Continue with the observation calculations for non-filtered samples
            # Calculate the weighted probabilities using probsArray and priors.
            # The multiplication of probsArray and priors represents a weighted version
            # of the probabilities, where probsArray contains the probabilities of events
            # and priors contains the corresponding weights.
            # This calculation is analogous to the odds ratio calculation in statistics,
            # where the odds ratio is defined as the ratio of the odds of two events.
            # The odds ratio captures the relationship between the weighted probabilities.
            # Specifically, the odds ratio is given by:
            #   OddsRatio = (P(A) * (1 - P(B))) / (P(B) * (1 - P(A)))
            # where P(A) and P(B) are the weighted probabilities of events A and B, respectively.
            odds = probsArray * priors
            observations = np.argmax(odds, axis=1)
            # Check if any element in the row is -1
            any_minus_one = np.any(probsArray == -1, axis=1)
            observations[any_minus_one] = -1
            observations_list.append(observations)

    # Concatenate observations for non-filtered samples
    observations = np.vstack(observations_list)

    # Transpose the observations array
    observations = observations.transpose()

    return observations


#############################
# countCNObservations
# Count the observations of copy number (CN) states for each patient and CN type,
# and calculate the transition matrices.
#
# Args:
# - observations (np.ndarray[int]): Array of shape (nbExons, nbSamples) containing the observed CN states
#     for each exon and sample.
# - nbStates [int]: Number of CN states.
#
# Returns a tuple (countArray, transitionMatrices):
# - countArray (np.ndarray[ints]): the count of each CN type for each sample, dim = NBsamps x NBCNStates
# - transitionMatrices (list of np.ndarray[ints]): transition matrices, each of shape (nbStates, nbStates),
#                                                  representing the transitions between CN types for each sample.
def countCNObservations(observations, nbStates):
    nbExons, nbSamples = observations.shape

    # Array to store the count of each CN type for each sample
    countArray = np.zeros((nbSamples, nbStates), dtype=int)

    # List to store the transition matrices
    transitionMatrices = []

    for sampleIndex in range(nbSamples):
        if np.all(observations[:, sampleIndex] != -1):
            continue

        # Count occurrences of each CN type, excluding -1 values
        valid_observations = observations[:, sampleIndex][observations[:, sampleIndex] != -1]
        counts = np.bincount(valid_observations, minlength=nbStates)
        countArray[sampleIndex] = counts

        # Calculate transition matrix
        countMatrix = np.zeros((nbStates, nbStates), dtype=int)

        prevCN = 2
        for exonIndex in range(nbExons):
            currCN = observations[exonIndex, sampleIndex]
            if currCN == -1:
                continue
            countMatrix[prevCN, currCN] += 1
            prevCN = currCN

        transitionMatrices.append(countMatrix)

    return (countArray, transitionMatrices)


######################################################
# mergeTransMatrices
# Merges a list of transition matrices by taking the sum of each element in the list.
# The resulting matrix is then normalized, ensuring that the sum of each row equals 1.
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

    return (normalized_arr)
