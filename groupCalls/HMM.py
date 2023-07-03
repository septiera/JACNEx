import logging
import numpy as np
import scipy.stats
import time

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
###############################################################################
# exonOnChr:
# Iterates over the exons and assigns a chromosome count to each exon based on
# the order in which the chromosomes appear.
# It keeps track of the previous chromosome encountered and increments the count
# when a new chromosome is found.
#
# Arg:
#  - a list of exons, each exon is a list of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID
#
# Returns an uint8 numpy.ndarray of the same size as exons, value is
# chromosome count for the corresponding exon.
def exonOnChr(exons):
    chrCounter = 0
    exon2Chr = np.zeros(len(exons), dtype=np.uint8)
    prevChr = exons[0][0]

    for idx, exon in enumerate(exons):
        currentChr = exon[0]

        if currentChr != prevChr:
            prevChr = currentChr
            chrCounter += 1

        exon2Chr[idx] = chrCounter

    return exon2Chr


#######################################
# HMM
# Hidden Markov Model (HMM) function for processing exon data.
#
# Args:
# - likelihoodMatrix (np.ndarray[floats]): pseudo emission probabilities (likelihood) of each state for each observation
#                                          for one sample.
#                                          dim = [NbStates, NbObservations]
# - exons (list of lists[str, int, int, str]): A list of exon information [CHR,START,END, EXONID]
# - transitionMatrix (np.ndarray[floats]): transition probabilities between states. dim = [NbStates, NbStates].
# - priors (numpy.ndarray): Prior probabilities.
#
# Returns:
#     numpy.ndarray: The path obtained from Viterbi algorithm.
def HMM(likelihoodMatrix, exons, transitionMatrix, priors):
    # Create a boolean mask for non-called exons [-1]
    exNotCalled = np.any(likelihoodMatrix == -1, axis=0)

    # Create a numpy array associating exons with chromosomes (counting)
    exon2Chr = exonOnChr(exons)

    # Initialize the path array with -1
    path = np.full(len(exons), -1, dtype=np.int8)

    # Iterate over each chromosome
    for thisChr in range(exon2Chr[-1] + 1):
        # Create a boolean mask for exons called on this chromosome
        exonsCalledThisChr = np.logical_and(~exNotCalled, exon2Chr == thisChr)
        # Get the path for exons called on this chromosome using Viterbi algorithm
        getPathThisChr = viterbi(likelihoodMatrix[exonsCalledThisChr], transitionMatrix, priors)
        # Assign the obtained path to the corresponding exons
        path[exonsCalledThisChr] = getPathThisChr

    return path


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# viterbi
# Implements the Viterbi algorithm to compute the most likely sequence of hidden states given observed emissions.
# It initializes the dynamic programming matrix, propagates scores through each observation, and performs
# backtracking to find the optimal state sequence. Additionally, it incorporates a weight ponderation based
# on the length of exons.
# This weight is applied to transitions between exons, favoring the normal number of copies (CN2) and reducing
# the impact of transitions between distant exons.
#
# Args:
# - CNcallOneSamp (np.ndarray[floats]): pseudo emission probabilities (log10_likelihood) of each state for each observation
#                                       for one sample
#                                       dim = [NbObservations, NbStates]
# - priors (np.ndarray[floats]): initial probabilities of each state
# - transMatrix (np.ndarray[floats]): transition probabilities between states. dim = [NbStates, NbStates]
# - exons (list of lists[str, int, int, str]): A list of exon information [CHR,START,END, EXONID]
#
# Returns:
# - bestPath (list[int]): the most likely sequence of hidden states given the observations and the HMM parameters.
def viterbi(CNcallOneSamp, priors, transMatrix, exons):
    # Fixed parameters
    # constant value used to normalise the distance between exons
    expectedCNVLength = 1.e8
    # Get the dimensions of the input matrix
    NbObservations, NbStates = CNcallOneSamp.shape

    # Transpose the input matrix in the same format as the emission matrix in the classic Viterbi algorithm
    # dim = [NbStates * NbObservations]
    CNcallOneSamp = CNcallOneSamp.transpose()

    # Step 1: Initialize variables
    # The algorithm starts by initializing the dynamic programming matrix "pathProbs" and the "path" matrix.
    # pathProbs: stores the scores of state sequences up to a certain observation,
    # path: is used to store the indices of previous states with the highest scores.
    pathProbs = np.full((NbStates, NbObservations), -np.inf, dtype=np.float128)
    path = np.zeros((NbStates, NbObservations), dtype=np.uint8)

    # Step 2: Fill the first column of path probabilities with prior values
    pathProbs[:, 0] = priors + CNcallOneSamp[:, 0]

    # Keep track of the end position of the previous exon
    previousExEnd = exons[0][2]

    # Step 3: Score propagation
    # Iterate through each observation, current states, and previous states.
    # Calculate the score of each state by considering:
    #  - the scores of previous states
    #  - transition probabilities (weighted by the distance between exons)
    for obNum in range(1, NbObservations):
        # Get the start position of the current exon and deduce the distance between the previous and current exons
        currentExStart = exons[obNum][1]
        exDist = currentExStart - previousExEnd

        for state in range(NbStates):
            # Decreasing weighting for transitions between exons as a function of their distance
            # - favoring the normal number of copies (CN2)
            # - no weighting for overlapping exons
            weight = 1.0
            if (state != 2) and (exDist > 0):
                weight = -exDist / expectedCNVLength

            probMax = -np.inf
            prevMax = -1

            for prevState in range(NbStates):
                pondTransition = (transMatrix[prevState, state] + weight)
                # Calculate the probability of observing the current observation given a previous state
                prob = pathProbs[prevState, obNum - 1] + pondTransition + CNcallOneSamp[state, obNum]
                # Find the previous state with the maximum probability
                # Update the value of the current state ("pathProbs")
                # Store the index of the previous state with the maximum probability in the "path" matrix
                if prob > probMax:
                    probMax = prob
                    prevMax = prevState

            # Store the maximum probability and CN type index for the current state and observation
            pathProbs[state, obNum] = probMax
            path[state, obNum] = prevMax

        previousExEnd = exons[obNum][2]

    # Step 4: Backtracking the path
    # Retrieve the most likely sequence of hidden states by backtracking through the "path" matrix.
    # Start with the state with the highest score in the last row of the "pathProbs" matrix.
    # Follow the indices of previous states stored in the "path" matrix to retrieve the previous state at each observation.
    bestPath = np.full(NbObservations, -1, dtype=np.uint8)
    bestPath[NbObservations - 1] = pathProbs[:, NbObservations - 1].argmax()
    for obNum in range(NbObservations - 1, 0, -1):
        bestPath[obNum - 1] = path[bestPath[obNum], obNum]

    return bestPath
