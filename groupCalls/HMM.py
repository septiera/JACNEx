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


#######################################
# logHMM
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
def logHMM(logLikelihood, exons, logTrans, logPriors):
    # Create a boolean mask for non-called exons [-1]
    exNotCalled = np.any(logLikelihood == -1, axis=0)

    # Create a numpy array associating exons with chromosomes (counting)
    exon2Chr = exonOnChr(exons)

    # Initialize the path array with -1
    path = np.full(len(exons), -1, dtype=np.int8)

    # Iterate over each chromosome
    for thisChr in range(exon2Chr[-1] + 1):
        # Create a boolean mask for exons called on this chromosome
        exonsCalledThisChr = np.logical_and(~exNotCalled, exon2Chr == thisChr)
        # Get the path for exons called on this chromosome using Viterbi algorithm
        getPathThisChr = logViterbi(logLikelihood[exonsCalledThisChr], logTrans, logPriors)
        # Assign the obtained path to the corresponding exons
        path[exonsCalledThisChr] = getPathThisChr

    return path


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# viterbi
# Implements the Viterbi algorithm to compute the most likely sequence of hidden states given the observed emissions.
# It initializes the dynamic programming matrix, performs the scaling step to avoid numerical issues, propagates
# the scores through each observation, scales the scores, and finally backtracks to find the most likely sequence of states.
#
# Args:
# - ProbsMatrix (np.ndarray[floats]): pseudo emission probabilities (likelihood) of each state for each observation.
#                                         dim = [NbStates, NbObservations]
# - priorProbs (np.ndarray[floats]): initial probabilities of each state
# - transitionMatrix (np.ndarray[floats]): transition probabilities between states. dim = [NbStates, NbStates]
#
# Returns:
# - seq (list[int]): the most likely sequence of hidden states given the observations and the HMM parameters.
def viterbi(ProbsMatrix, priorProbs, transitionMatrix):
    ProbsMatrix = ProbsMatrix.transpose()
    # Get the number of time steps (T) and the number of hidden states (N)
    NbStates, NbObservations = ProbsMatrix.shape

    statusPrint = True  # DEBUG initialize statusPrint flag for printing status

    # Step 1: Initialization
    # Initialize the pathProbs and path matrices
    pathProbs = np.zeros((NbStates, NbObservations), dtype=np.float128)
    path = np.zeros((NbStates, NbObservations), dtype=np.uint8)

    # Step 2: Initialization of the first observation
    # Calculate the initial path probabilities using prior probabilities and sample probabilities
    pathProbs[:, 0] = priorProbs * ProbsMatrix[:, 0]

    # Step 3: Score propagation
    # Calculate the path probabilities for each observation and state
    for obNum in range(1, NbObservations):
        for state in range(NbStates):
            prevMax = -1
            probMax = 0
            for prevState in range(NbStates):
                prob = pathProbs[prevState, obNum - 1] * transitionMatrix[prevState, state] * ProbsMatrix[state, obNum]
                if prob > probMax:
                    probMax = prob
                    prevMax = prevState
                    pathProbs[state, obNum] = probMax
                    path[state, obNum] = prevMax

        #### DEBUG part
        if np.sum(pathProbs[:, obNum]) == 0:
            if statusPrint:
                logger.debug(obNum, pathProbs[:, obNum - 1], pathProbs[:, obNum])
                statusPrint = False

    # Step 4: Backtracking the path
    # Retrieve the best path by backtracking through the path matrix
    bestPath = np.full(NbObservations, -1, dtype=np.uint8)
    bestPath[NbObservations - 1] = np.argmax(pathProbs[:, NbObservations - 1])
    for obNum in range(NbObservations - 1, 0, -1):
        bestPath[obNum - 1] = path[bestPath[obNum], obNum]

    return bestPath


######################################
# logViterbi
# Implements the Viterbi algorithm to compute the most likely sequence of hidden states given the observed emissions.
# It initializes the dynamic programming matrix, performs the scaling step to avoid numerical issues, propagates
# the scores through each observation, scales the scores, and finally backtracks to find the most likely sequence of states.
#
# Args:
# - LogProbsMatrix (np.ndarray[floats]): pseudo emission probabilities (Log likelihood) of each state for each observation.
#                                        dim = [NbStates, NbObservations]
# - priorProbs (np.ndarray[floats]): initial probabilities of each state
# - transitionMatrix (np.ndarray[floats]): transition probabilities between states. dim = [NbStates, NbStates]
#
# Returns:
# - seq (list[int]): the most likely sequence of hidden states given the observations and the HMM parameters.
def logViterbi(LogProbsMatrix, priorProbs, transitionMatrix):
    LogProbsMatrix = LogProbsMatrix.transpose()
    # Get the number of time steps (T) and the number of hidden states (N)
    NbStates, NbObservations = LogProbsMatrix.shape

    # Step 1: Initialization
    # Initialize the pathProbs and path matrices
    pathProbs = np.full((NbStates, NbObservations), -np.inf, dtype=np.float128)
    path = np.zeros((NbStates, NbObservations), dtype=np.uint8)

    # Step 2: Initialization of the first observation
    # Calculate the initial path probabilities using log-transformed sample probabilities and non-log transformed priors
    pathProbs[:, 0] = priorProbs + LogProbsMatrix[:, 0]

    # Step 3: Score propagation
    # Calculate the path probabilities for each observation and state
    for obNum in range(1, NbObservations):
        for state in range(NbStates):
            prevMax = -1
            probMax = -np.inf
            for prevState in range(NbStates):
                prob = pathProbs[prevState, obNum - 1] + transitionMatrix[prevState, state] + LogProbsMatrix[state, obNum]
                if prob > probMax:
                    probMax = prob
                    prevMax = prevState
                    pathProbs[state, obNum] = probMax
                    path[state, obNum] = prevMax

    # Step 4: Backtracking the path
    # Retrieve the best path by backtracking through the path matrix
    bestPath = np.full(NbObservations, -1, dtype=np.uint8)
    bestPath[NbObservations - 1] = np.argmax(pathProbs[:, NbObservations - 1])
    for obNum in range(NbObservations - 1, 0, -1):
        bestPath[obNum - 1] = path[bestPath[obNum], obNum]

    return bestPath
