import concurrent.futures
import logging
import math
import numpy

####### JACNEx modules
import callCNVs.transitions
import callCNVs.exonDistance

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
######################################
# applyHMM
# Processes CNV calls for a given set of samples in parallel using the HMM Viterbi algorithm.
# Args:
# - samples (list[strs]): List of sample identifiers.
# - autosomeExons (list[str, int, int, str]): exon on autosome infos [chr, START, END, EXONID].
# - gonosomeExons (list[str, int, int, str]): exon on gonosome infos.
# - likelihoods_A (dict): key==sample ID, value==Likelihoods for autosomal chromosomes,
#                         numpy.ndarray 2D [floats], dim = NbofExons * NbOfCNStates
# - likelihoods_G (dict): key==sample ID, value==Likelihoods for gonosomal chromosomes
# - transMatrix (numpy.ndarray[floats]): Transition matrix for the HMM Viterbi algorithm.
# - jobs (int): Number of jobs to run in parallel.
# - dmax (int): Maximum distance threshold between exons.
#
# Returns a tuple of two lists: The first list contains CNV information for autosomal chromosomes,
# and the second list for gonosomal chromosomes. Each list contains tuples with CNV information:
# [CNType, exonIndexStart, exonIndexEnd, bestPathProbabilities, sampleName].
def applyHMM(samples, autosomeExons, gonosomeExons, likelihoods_A, likelihoods_G, transMatrix, jobs, dmax):
    CNVs_A = []
    CNVs_G = []
    paraSample = min(math.ceil(jobs / 2), len(samples))
    logger.info("%i samples => will process %i in parallel", len(samples), paraSample)

    with concurrent.futures.ProcessPoolExecutor(paraSample) as pool:
        processSamps(samples, autosomeExons, likelihoods_A, transMatrix, pool, CNVs_A, dmax)
        processSamps(samples, gonosomeExons, likelihoods_G, transMatrix, pool, CNVs_G, dmax)
    return (CNVs_A, CNVs_G)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# processSamps
# Processes a specific type of chromosome (either autosomal or gonosomal) for CNV calls.
# This function iterates over each sample to submit CNV calling tasks to a multiprocessing pool.
#
# Args:
# - samples (list[strs]): A list of sample identifiers.
# - exons (list[str, int, int, str]): exon on autosome infos [chr, START, END, EXONID].
# - likelihoods (dict): key==sample ID, value==Likelihoods,
#                       numpy.ndarray 2D [floats], dim = NbofExons * NbOfCNStates
# - transMatrix (numpy.ndarray[floats]): A transition matrix used in the HMM Viterbi algorithm.
# - pool (concurrent.futures.Executor): A concurrent executor for parallel processing.
# - CNVs (list[str, int, int, int, floats, str]): CNV infos [CNType, exonStart, exonEnd, pathProb, sampleName]
# - dmax (int): Maximum distance threshold between exons.
def processSamps(samples, exons, likelihoods, transMatrix, pool, CNVs, dmax):
    for sampID in samples:
        # check if the sample ID is present in the likelihoods dictionary
        if sampID not in likelihoods.keys():
            logger.debug("no CNV calling for sample %s", sampID)
            continue
        # submit a task for processing the chromosome data for the current sample
        # task is submitted to the provided process pool for parallel execution
        futureRes = pool.submit(viterbi(likelihoods[sampID], transMatrix, sampID, exons, dmax))
        # add a callback to the future object
        # once the task is complete, the concatCNVs function will be called with the result
        # the concatCNVs function will handle the aggregation of CNVs from the result
        futureRes.add_done_callback(lambda future: concatCNVs(future, CNVs))


######################################
# concatCNVs
# A callback function for processing the result of a Viterbi algorithm task.
# Extracts the result from the Future object and appends it to a global CNVs list.
#
# Args:
# - futureSampCNVExtract (concurrent.futures.Future): A Future object for an
#    asynchronous Viterbi task.
# - CNVs (list): Global list to which the results are appended.
#    Each element is a tuple representing CNV information.

# No return value; updates the CNVs list in place.
def concatCNVs(futureSampCNVExtract, CNVs):
    e = futureSampCNVExtract.exception()
    if e is not None:
        logger.warning("Failed viterbi %s", str(e))
    else:
        viterbiRes = futureSampCNVExtract.result()
        countCNVs(viterbiRes)
        # append each CNV from the result to the global CNVs list
        for cnv in range(len(viterbiRes)):
            CNVs.append(viterbiRes[cnv])


######################################
# countCNVs
# Counts the occurrences of each CNV type called for a sample and logs the result.
#
# Args:
# - sampCNVs (list of lists): CNV data for a sample. Each inner list contains CNV information.
def countCNVs(sampCNVs):
    # Initialize a dictionary to count occurrences of each CN
    cn_counts = {}
    for record in sampCNVs:
        cn = record[0]
        cn_counts[cn] = cn_counts.get(cn, 0) + 1

    # Prepare the string for each CN and its count, including CNs with 0 occurrences
    if len(cn_counts.keys()) != 0:
        cn_list = [f"CN{cn}:{cn_counts.get(cn, 0)}" for cn in range(max(cn_counts.keys()) + 1)]
        cn_str = ', '.join(cn_list)
        logger.debug("Done callCNvs for %s, %s", sampCNVs[0][4], cn_str)


######################################
# viterbi
# implements the Viterbi algorithm, a dynamic programming approach, for Hidden Markov
# Models(HMMs) in the context of identifying CNVs.
# objective is to find the most likely sequence of hidden states (CNVs) given a sequence
# of observations (chromosome likelihoods) and the transition probabilities between states.
#
# Specs:
# - Initialization: initializing necessary variables, including the probability matrices
#    and the path matrix.
# - Adjustment for Exon Distance: adjusts transition probabilities based on the distance
#   between exons, employing a power law approach to smooth probabilities as the distance increases
#   up to a maximum distance (dmax).
# - Score Propagation: iterates over each observation (exon) and calculates the probabilities
#    for each state by considering the likelihood of the current observation and the transition
#    probabilities from previous states.
# - Backtracking and Resetting: special condition where it backtracks and resets probabilities
#    under certain circumstances (e.g., when all live states have the same predecessor state).
# - Path Tracking and CNV Aggregation: tracks the most probable path for each state and observation.
#    aggregates the CNV calls by backtracking through the path matrix at the end of the observations.
# - Error Handling: includes error handling to log and raise exceptions in case of inconsistencies
#    or failures during the process.
#
# Args:
# - likelihoods (numpy.ndarray[floats]): pseudo-emission probabilities (likelihood) of each
#    state for each observation (exon) for one sample. Dim = [NbObservations, NbStates].
# - transMatrix (numpy.ndarray[floats]): transition probabilities between states, including a
#    void status. Dim = [NbStates + 1, NbStates + 1].
# - sampleID [str]
# - exons [list of lists[str, int, int, str]]: exon infos [chr, START, END, EXONID].
# - dmax [int]: Maximum distance threshold between exons.
#
# Returns:
# - sampCNVs (list of list[int, int, int, floats, str]): A list of CNVs detected. Each CNV is
#    represented as a list containing the CNV type, start exon index, end exon index, path probability,
#    and sample ID.
def viterbi(likelihoods, transMatrix, sampleID, exons, dmax):
    try:
        CNVs = []

        # get the dimensions of the input matrix
        NbObservations = likelihoods.shape[0]
        NbStatesWithVoid = len(transMatrix)

        # control that the right number of hidden state
        if NbStatesWithVoid != likelihoods.shape[1] + 1:
            logger.error("NbStates not consistent with number of columns + 1 in chromCalls")
            raise

        # Transpose the input matrix in the same format as the emission matrix in the classic
        # Viterbi algorithm, dim = [NbStates * NbObservations]
        likelihoods = likelihoods.transpose()

        # Step 1: Initialize variables
        # probsPrev[i]: stores the probability of the most likely path ending in state i at the previous exon
        probsPrev = numpy.zeros(NbStatesWithVoid, dtype=numpy.float128)
        probsPrev[0] = 1
        # probsCurrent[i]: same ending at current exon
        probsCurrent = numpy.zeros(NbStatesWithVoid, dtype=numpy.float128)
        # path[i,e]: state at exon e-1 that produces the highest probability ending in state i at exon e
        # (ie, probsCurrent[i])
        # this state is 0 (void) if the path starts here
        path = numpy.zeros((NbStatesWithVoid, NbObservations), dtype=numpy.uint8)

        # Step 2: Score propagation
        # Iterate through each observation, current states, and previous states.
        # Calculate the score of each state by considering:
        #  - the scores of previous states
        #  - transition probabilities
        prevChrom = exons[0][0]
        prevEnd = exons[0][2]
        lastCalledExInd = -1
        for exonIndex in range(len(exons)):
            currentChrom = exons[exonIndex][0]
            currentStart = exons[exonIndex][1]
            currentEnd = exons[exonIndex][2]

            if currentChrom != prevChrom:
                tmpList = backtrack_aggregateCalls(path, lastCalledExInd, probsPrev.argmax(), max(probsPrev), sampleID)
                CNVs.extend(tmpList)
                # reinit
                prevChrom = currentChrom
                prevEnd = exons[exonIndex][2]
                probsPrev[0] = 1
                probsPrev[1:] = 0
                path[0, exonIndex - 1] = 0
                path[1:, exonIndex - 1] = 5

            if likelihoods[0, exonIndex] == -1:
                # no call, skip
                continue

            # adjusts transition probabilities.
            distFromPrevEx = currentStart - prevEnd - 1
            adjustedTransMatrix = callCNVs.exonDistance.adjustTransMatrix(transMatrix, distFromPrevEx, dmax)

            for currentState in range(NbStatesWithVoid):
                print("#### current", currentState)
                probMax = -1
                prevStateMax = -1

                # state 0 == void is never in a path except at initialisation/reset => keep defaults
                if currentState == 0:
                    probsCurrent[currentState] = 0
                    path[currentState, exonIndex] = 0
                    continue

                else:
                    for prevState in range(NbStatesWithVoid):
                        # calculate the probability of observing the current observation given a previous state
                        prob = (probsPrev[prevState] *
                                adjustedTransMatrix[prevState, currentState] *
                                likelihoods[currentState - 1, exonIndex])
                        # currentState - 1 because chromLikelihoods doesn't have values for void state

                        print(exonIndex, prevState, probsPrev[prevState],
                              adjustedTransMatrix[prevState, currentState],
                              likelihoods[currentState - 1, exonIndex], prob)
                        # Find the previous state with the maximum probability
                        if prob >= probMax:
                            probMax = prob
                            prevStateMax = prevState

                    # Store the maximum probability and CN type index for the current state and observation
                    probsCurrent[currentState] = probMax
                    path[currentState, exonIndex] = prevStateMax
            print("PROBCurrent", probsCurrent, "PATHSTATE", path[:, exonIndex])

            # if all LIVE states (probsCurrent > 0) at currentExon have the same
            # predecessor state and that state is CN2 : backtrack from [previous exon, CN2]
            # and reset
            if (((probsCurrent[1] == 0) or (path[1, exonIndex] == 3)) and
                ((probsCurrent[2] == 0) or (path[2, exonIndex] == 3)) and
                ((probsCurrent[3] == 0) or (path[3, exonIndex] == 3)) and
                ((probsCurrent[4] == 0) or (path[4, exonIndex] == 3))):
                print("RESET")
                try:
                    tmpList = backtrack_aggregateCalls(path, lastCalledExInd, 3, max(probsPrev), sampleID)
                    CNVs.extend(tmpList)
                    # logger.info("successfully backtrack from %i exInd", exonIndex - 1)
                except Exception:
                    logger.error("sample %s, %s, exon %i, exon -1 %i", sampleID, exonIndex, exonIndex - 1)
                    logger.error("%f, %f, %f, %f, %f", probsCurrent[0], probsCurrent[1],
                                 probsCurrent[2], probsCurrent[3], probsCurrent[4])
                    return (CNVs)

                # reset prevScores to (1,0,0,0,0), and loop again
                probsPrev[0] = 1
                probsPrev[1:] = 0
                path[0, exonIndex - 1] = 0
                path[1:, exonIndex - 1] = 5
                prevEnd = currentEnd
                continue
            else:
                # Update previous probabilities and move to the next exon
                for i in range(len(probsCurrent)):
                    probsPrev[i] = probsCurrent[i]

            lastCalledExInd = exonIndex
            prevEnd = currentEnd

        # Final backtrack to aggregate calls for the last exon
        tmpList = backtrack_aggregateCalls(path, lastCalledExInd, probsPrev.argmax(), max(probsPrev), sampleID)
        CNVs.extend(tmpList)

        return (CNVs)

    except Exception as e:
        logger.error("CNCalls failed for sample %s - error: %s - exonIndex %s", sampleID, repr(e), str(exonIndex))
        raise Exception(sampleID)


######################################
# backtrack_aggregateCalls
# retrieve the most probable sequence of hidden states (CNVs) by backtracking through
# the path matrix generated by the Viterbi algorithm.
# aggregates consecutive exons with the same predicted CN into a single CNV call.
#
# Specs:
# - Initialization: initializing necessary variables, including CNV lists and exon indices.
# - Reverse Iteration: iterates over the exons in reverse order, starting from the last
#    processed exon, and follows the path of states backward to the start of the chromosome.
# - CNV Aggregation and Special Cases Handling: aggregates consecutive exons with the same CN
#    state into a single CNV. It also handles special cases like transitions between different
#    CN states, and extends certain CNVs (like HETDEL) to the boundaries of other types (like HVDEL)
#    when necessary.
# - Final CNV List: aggregated CNVs are then compiled into a final list, which is reversed to match
#    the original chronological order of exons.
#
# Args:
# - path (numpy.ndarray[ints]): optimal state transitions for each time step, as determined by the
#    Viterbi algorithm. Dimensions are [NBState, NBObservations].
# - lastExon [int]: index of the last exon traversed by the Viterbi algorithm before a reset
#    or the end of the chromosome path.
# - lastState [int]: state with the best probability for the last exon, indicating the end of
#    the best path.
# - pathProb [float]: probability of the path ending in lastState.
# - sampleID [str]
#
# Returns a list of CNVs, a CNV == [CNType, startExon, endExon, pathProb, sampleID]
def backtrack_aggregateCalls(path, lastExon, lastState, pathProb, sampleID):
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
                CNVs.append([nextState, startExon, endExon, pathProb, sampleID])
                if hetDelEndExon is not None:
                    CNVs.append([2, startExon, hetDelEndExon, pathProb, sampleID])
                    hetDelEndExon = None
            break

        elif currentState == nextState:
            pass

        elif nextState == 3:
            endExon = nextExon - 1

        elif currentState == 3:
            startExon = nextExon
            CNVs.append([nextState, startExon, endExon, pathProb, sampleID])
            if hetDelEndExon is not None:
                CNVs.append([2, startExon, hetDelEndExon, pathProb, sampleID])
                hetDelEndExon = None

        # DUPs
        elif (currentState == 4) or (nextState == 4):
            startExon = nextExon
            CNVs.append([nextState, startExon, endExon, pathProb, sampleID])
            if hetDelEndExon is not None:
                CNVs.append([2, startExon, hetDelEndExon, pathProb, sampleID])
                hetDelEndExon = None
            endExon = nextExon - 1

        # Special cases:
        # If a HVDEL is followed by a HETDEL or vice versa,
        # extend the HETDEL to the boundaries of the HVDEL.
        # 1) HETDEL followed by HVDEL
        elif (currentState == 2) and (nextState == 1):
            # create HVDEL
            startExon = nextExon
            CNVs.append([nextState, startExon, endExon, pathProb, sampleID])
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


######################################
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
def mapCNVIndexToExons(CNVs, exIndexCalled, firstExOnChrom):
    for event in range(len(CNVs)):
        CNVs[event][0] = CNVs[event][0] - 1
        CNVs[event][1] = firstExOnChrom + exIndexCalled[CNVs[event][1]]
        CNVs[event][2] = firstExOnChrom + exIndexCalled[CNVs[event][2]]
