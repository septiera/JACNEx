import logging
import numpy as np
import concurrent.futures

import callCNVs.transitions

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
######################################
# processCNVCalls
# Processes CNV calls for a given set of samples in parallel.
# Args:
# - samples (list[strs]): List of sample identifiers.
# - autosomeExons (list[str, int, int, str]): exon on autosome infos [chr, START, END, EXONID].
# - gonosomeExons (list[str, int, int, str]): exon on gonosome infos.
# - likelihoods_A (dict): key==sample ID, value==Likelihoods for autosomal chromosomes,
#                         np.ndarray 2D [floats], dim = NbofExons * NbOfCNStates
# - likelihoods_G (dict): key==sample ID, value==Likelihoods for gonosomal chromosomes
# - transMatrix (np.ndarray[floats]): Transition matrix for the HMM Viterbi algorithm.
# - jobs (int): Number of jobs to run in parallel.
#
# Returns:
# - CNVs (list[str, int, int, int, floats, str]): CNV infos [chromType, CNType, exonStart, exonEnd, pathProb, sampleName]
def process_cnv_calls(samples, autosomeExons, gonosomeExons, likelihoods_A, likelihoods_G, transMatrix, jobs):
    CNVs = []
    paraSample = min(max(jobs // 2, 1), len(samples))
    logger.info("%i samples => will process %i in parallel", len(samples), paraSample)

    with concurrent.futures.ProcessPoolExecutor(paraSample) as pool:
        processChromType("A", samples, autosomeExons, likelihoods_A, transMatrix, pool, CNVs)
        processChromType("G", samples, gonosomeExons, likelihoods_G, transMatrix, pool, CNVs)

    return CNVs


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# concatCNVs
# A callback function for processing the result of a Viterbi algorithm task.
# This function is intended to be used as a callback for a Future object in
# a concurrent execution environment.
# It extracts the result from the Future object, processes it,
# and appends it to the global CNVs list.
#
# Args:
# futureViterbi (concurrent.futures.Future): A Future object representing an asynchronous execution
#                                            of the Viterbi algorithm.
# CNVs (list[str, int, int, int, floats, str]): A global list to which the results of the Viterbi
#                                               algorithm are appended.
def concatCNVs(futureViterbi, CNVs):
    e = futureViterbi.exception()
    if e is not None:
        logger.warning("Failed viterbi for sample %s, skipping it", str(e))
    else:
        viterbiRes = futureViterbi.result()
        CNVs.extend(viterbiRes)


######################################
# processChromType
# Processes a specific type of chromosome (either autosomal or gonosomal) for CNV calls.
# This function iterates over each sample and each exon range to submit CNV calling tasks
# to a multiprocessing pool.
# It leverages the Viterbi algorithm to identify CNVs for each chromosome type.

# Args:
# - chromType [str]: "A"==autosomes, "G"==gonosomes
# - samples (list[strs]): A list of sample identifiers.
# - exons (list[str, int, int, str]): exon on autosome infos [chr, START, END, EXONID].
# - likelihoods (dict): key==sample ID, value==Likelihoods,
#                       np.ndarray 2D [floats], dim = NbofExons * NbOfCNStates
# - transMatrix (np.ndarray[floats]): A transition matrix used in the HMM Viterbi algorithm.
# - pool (concurrent.futures.Executor): A concurrent executor for parallel processing.
# - CNVs (list[str, int, int, int, floats, str]): CNV infos [chromType, CNType, exonStart, exonEnd, pathProb, sampleName]

def processChromType(chromType, samples, exons, likelihoods, transMatrix, pool, CNVs):
    isFirstExon = callCNVs.transitions.flagChromStarts(exons)
    for sampID in samples:
        for i, firstExOnChrom in enumerate(isFirstExon):
            if i + 1 < len(isFirstExon):
                lastExOnChrom = isFirstExon[i + 1] - 1
            else:
                lastExOnChrom = len(exons)
            sampLikelihoodOneChrom = likelihoods[sampID][firstExOnChrom:lastExOnChrom, :]
            futureRes = pool.submit(viterbi, chromType, sampLikelihoodOneChrom, transMatrix, sampID)
            futureRes.add_done_callback(lambda future: concatCNVs(future, CNVs))


######################################
# viterbi
# Implements the Viterbi algorithm for Hidden Markov Models.
# Given likelihoods for observations and transition probabilities, it finds
# the most likely hidden state sequence.
# The function returns a list of CNVs represented as [CNType, startExon, endExon].
# It calculates probabilities and paths, handles resets, and aggregates CNVs.
# The process involves backtracking and path tracking.
#
# Args:
# - chromType [str]: "A"==autosomes, "G"==gonosomes
# - CNcallOneSamp (np.ndarray[floats]): pseudo emission probabilities (likelihood)
#                                       of each state for each observation for one sample
#                                       dim = [NbObservations, NbStates]
# - transMatrix (np.ndarray[floats]): transition probabilities between states + void status
#                                     dim = [NbStates +1, NbStates + 1]
# - sampleID [str]
#
# Returns:
# - sampCNVs (list[str, int, int, int, floats, str]): sample CNV infos [chromType, CNType,
#                                                     exonStart, exonEnd, pathProb, sampleID]
def viterbi(chromType, CNCallOneSamp, transMatrix, sampleID):
    try:
        # list of lists to return
        # First filled with [Cntype, startExonCalledIndex, endExonCalledIndex],
        # and then with [Cntype, startExonIndex, endExonIndex].
        CNVs = []

        # np.ndarray of called exon indexes
        exIndexCalled = np.where(np.all(CNCallOneSamp != -1, axis=1))[0]

        # Get the dimensions of the input matrix
        NbObservations = len(exIndexCalled)
        NbStatesWithVoid = len(transMatrix)

        # control that the right number of hidden state
        if NbStatesWithVoid != CNCallOneSamp.shape[1] + 1:
            logger.error("NbStates not consistent with number of columns + 1 in CNcallOneSamp")
            raise

        # Transpose the input matrix in the same format as the emission matrix in the classic
        # Viterbi algorithm, dim = [NbStates * NbObservations]
        CNCallOneSamp = CNCallOneSamp.transpose()

        # Step 1: Initialize variables
        # probsPrev[i]: stores the probability of the most likely path ending in state i at the previous exon
        probsPrev = np.zeros(NbStatesWithVoid, dtype=np.float128)
        probsPrev[0] = 1
        # probsCurrent[i]: same ending at current exon
        probsCurrent = np.zeros(NbStatesWithVoid, dtype=np.float128)
        # path[i,e]: state at exon e-1 that produces the highest probability ending in state i at exon e
        # (ie, probsCurrent[i])
        # this state is 0 (void) if the path starts here
        path = np.zeros((NbStatesWithVoid, NbObservations), dtype=np.uint8)

        # Step 2: Score propagation
        # Iterate through each observation, current states, and previous states.
        # Calculate the score of each state by considering:
        #  - the scores of previous states
        #  - transition probabilities
        exonIndex = 0
        while exonIndex < len(exIndexCalled):

            for currentState in range(NbStatesWithVoid):
                probMax = -1
                prevStateMax = -1

                # state 0 == void is never in a path except at initialisation/reset => keep defaults
                if currentState == 0:
                    probsCurrent[currentState] = 0
                    path[currentState, exonIndex] = 0
                    continue

                else:
                    for prevState in range(NbStatesWithVoid):

                        # Calculate the probability of observing the current observation given a previous state
                        prob = (probsPrev[prevState] *
                                transMatrix[prevState, currentState] *
                                CNCallOneSamp[currentState - 1, exIndexCalled[exonIndex]])
                        # currentState - 1 because CNCallOneSamp doesn't have values for void state

                        # Find the previous state with the maximum probability
                        # Store the index of the previous state with the maximum probability in the "path" matrix
                        if prob >= probMax:
                            probMax = prob
                            prevStateMax = prevState

                    # Store the maximum probability and CN type index for the current state and observation
                    probsCurrent[currentState] = probMax
                    path[currentState, exonIndex] = prevStateMax

            # if all LIVE states (probsCurrent > 0) at currentExon have the same
            # predecessor state and that state is CN2 : backtrack from [previous exon, CN2]
            # and reset
            if (((probsCurrent[1] == 0) or (path[1, exonIndex] == 3)) and
                ((probsCurrent[2] == 0) or (path[2, exonIndex] == 3)) and
                ((probsCurrent[3] == 0) or (path[3, exonIndex] == 3)) and
                ((probsCurrent[4] == 0) or (path[4, exonIndex] == 3))):

                try:
                    tmpList = backtrack_aggregateCalls(path, exonIndex - 1, 3, probMax, chromType, sampleID)
                    CNVs.extend(tmpList)
                    # logger.info("successfully backtrack from %i exInd", exonIndex - 1)
                except Exception:
                    logger.error("sample %s, %s, exon %i, exon -1 %i", sampleID, chromType, exIndexCalled[exonIndex], exIndexCalled[exonIndex - 1])
                    logger.error("%f, %f, %f, %f, %f", probsCurrent[0], probsCurrent[1],
                                 probsCurrent[2], probsCurrent[3], probsCurrent[4])
                    return (sampleID, CNVs)

                # reset prevScores to (1,0,0,0,0), and loop again (WITHOUT incrementing exonIndex)
                probsPrev[1:] = 0
                probsPrev[0] = 1
                path[0, exonIndex - 1] = 0
                path[1:, exonIndex - 1] = 5
                continue

            else:
                for i in range(len(probsCurrent)):
                    probsPrev[i] = probsCurrent[i]
                exonIndex += 1

        tmpList = backtrack_aggregateCalls(path, exonIndex - 1, probsPrev.argmax(), probMax)
        CNVs.extend(tmpList)

        mapCNVIndexToExons(CNVs, exIndexCalled)

        return (CNVs)

    except Exception as e:
        logger.error("CNCalls failed for sample n°%s - %s", sampleID, repr(e))
        raise Exception(sampleID)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
################################
# backtrack_aggregateCalls
# Retrieve the most likely sequence of hidden states by backtracking through the "path" matrix.
# aggregation of exons if same CN predicted in CNVs.
#
# Args:
# - path (np.ndarray[uint8]): dynamic matrix generated by the Viterbi algorithm, used to store
#                             the optimal state transitions for each time step.
#                             dim = [NBState, NBObservations]
# - lastExon [int]: last exon traversed by the viterbi algorithm before the reset or the end
#                   of the chromosome path.
# - lastState [int]: state with the best probability for the last exon, best path.
#
# Returns a list of CNVs, a CNV == [chromType, CNType, startExon, endExon, pathProb, sampleID]
def backtrack_aggregateCalls(path, lastExon, lastState, pathProb, chromType, sampleID):
    CNVs = []
    startExon = None
    endExon = None
    # retention of an exon index to correctly delimit hetDELs in special cases
    # (see following description in the code)
    hetDelEndExon = None

    # Iterate over exons in reverse order
    nextExon = lastExon
    nextState = lastState

    if nextState != 3:
        endExon = nextExon

    while True:
        try:
            currentState = path[nextState, nextExon]
        except Exception:
            logger.error("nextEx=%i, lastEx=%i, lastState=%i", nextExon, lastExon, lastState)
            toPrint = ""
            for i in range(lastExon - 10, lastExon + 1):
                for j in range(5):
                    toPrint += "\t" + str(path[j, i])
                toPrint += "\n"
            logger.error(toPrint)
            raise

        # currentState == 0, we are at the beginning of the best path,
        # create first CNV(s) if needed
        if currentState == 0:
            if nextState != 3:
                startExon = nextExon
                CNVs.append([chromType, nextState, startExon, endExon, pathProb, sampleID])
                if hetDelEndExon is not None:
                    CNVs.append([chromType, 2, startExon, hetDelEndExon, pathProb, sampleID])
                    hetDelEndExon = None
            break

        elif currentState == nextState:
            pass

        elif nextState == 3:
            endExon = nextExon - 1

        elif currentState == 3:
            startExon = nextExon
            CNVs.append([chromType, nextState, startExon, endExon, pathProb, sampleID])
            if hetDelEndExon is not None:
                CNVs.append([chromType, 2, startExon, hetDelEndExon, pathProb, sampleID])
                hetDelEndExon = None

        # DUPs
        elif (currentState == 4) or (nextState == 4):
            startExon = nextExon
            CNVs.append([chromType, nextState, startExon, endExon, pathProb, sampleID])
            if hetDelEndExon is not None:
                CNVs.append([chromType, 2, startExon, hetDelEndExon, pathProb, sampleID])
                hetDelEndExon = None
            endExon = nextExon - 1

        # Special cases:
        # If a HVDEL is followed by a HETDEL or vice versa,
        # extend the HETDEL to the boundaries of the HVDEL.
        # 1) HETDEL followed by HVDEL
        elif (currentState == 2) and (nextState == 1):
            # create HVDEL
            startExon = nextExon
            CNVs.append([chromType, nextState, startExon, endExon, pathProb, sampleID])
            # keep endExon except if HVDEL was preceded by another HETDEL,
            # in which case ...
            if hetDelEndExon is not None:
                endExon = hetDelEndExon
                hetDelEndExon = None

        # 2) HVDEL followed by HETDEL
        elif (currentState == 1) and (nextState == 2):
            hetDelEndExon = endExon
            endExon = nextExon - 1
        else:
            raise Exception("not captured case")

        nextState = currentState
        nextExon -= 1

    CNVs.reverse()

    return(CNVs)


############
# mapCNVIndexToExons
# Map CNV indexes to "exons" indexes based on the 'exIndexCalled' list.
#
# Args:
# CNVs (list of lists): A list containing CNV events, where each event is represented
#                       as a list [start, end, state].
# exIndexCalled (list[int]): A list containing the indexes of exons used in the Viterbi
#                            algorithm's output.
#
# Returns:
# None (The CNVs list will be modified in-place.)
def mapCNVIndexToExons(CNVs, exIndexCalled):
    for event in range(len(CNVs)):
        CNVs[event][0] = CNVs[event][0] - 1
        CNVs[event][1] = exIndexCalled[CNVs[event][1]]
        CNVs[event][2] = exIndexCalled[CNVs[event][2]]
