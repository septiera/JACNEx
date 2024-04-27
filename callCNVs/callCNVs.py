import concurrent.futures
import logging
import math
import numpy

####### JACNEx modules
import callCNVs.transitions

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
        NbStatesWithVoid = len(transMatrix)

        # sanity
        if NbStatesWithVoid != likelihoods.shape[1] + 1:
            logger.error("NbStates not consistent with number of columns + 1 in chromCalls")
            raise

        # Step 1: Initialize variables
        # probsPrev[s]: probability of the most likely path ending in state s at the previous exon
        probsPrev = numpy.zeros(NbStatesWithVoid, dtype=numpy.float128)
        probsPrev[0] = 1
        # probsCurrent[s]: same ending at current exon
        probsCurrent = numpy.zeros(NbStatesWithVoid, dtype=numpy.float128)
        # chrom and end of previous exon
        prevChrom = exons[0][0]
        prevEnd = 0

        # temp data structures used by buildCNVs() and reset whenever it is called:
        # list of indexes in "exons"
        calledExons = []
        # path[e][s]: state at exon e-1 (in calledExons) that produces the highest
        # probability ending in state s at exon e (ie, probsCurrent[s])
        path = []
        # bestPathProbas[e][s] == proba of most likely path starting at calledExons[0]
        # (state 0 or 3) and ending in state s at exon calledExons[e] - this is populated
        # using probsCurrent
        bestPathProbas = []
        # CN2PathProbas[i] == proba of path starting at calledExons[0][3] and staying
        # in state 3==CN2 all along
        CN2PathProbas = []

        # Step 2: viterbi forward algorithm
        for exonIndex in range(len(exons)):
            if exonIndex > 300:
                break

            if likelihoods[exonIndex, 0] == -1:
                # exon is no-call => skip
                continue

            if exons[exonIndex][0] != prevChrom:
                if len(calledExons) > 0:
                    print("RESETCHROM")
                    CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2PathProbas,
                                          bestPathProbas[-1].argmax(), sampleID))
                # reinit
                probsPrev[0] = 1
                probsPrev[1:] = 0
                prevChrom = exons[exonIndex][0]
                prevEnd = 0
                calledExons = []
                path = []
                bestPathProbas = []
                CN2PathProbas = []

            # adjust transition probabilities except for first called exon on a chrom
            if prevEnd != 0:
                distFromPrevEx = exons[exonIndex][1] - prevEnd - 1
                adjustedTransMatrix = callCNVs.exonDistance.adjustTransMatrix(transMatrix, distFromPrevEx, dmax)
            else:
                adjustedTransMatrix = transMatrix

            # calculate proba of the most likely paths ending in each state for current exon
            # state 0 == void is never in a path except at initialisation/reset
            probsCurrent[0] = 0

            # accumulators for populating the data structures for buildCNVs(), which will only
            # be populated after possibly calling buildCNVs()
            bestPrevState = numpy.zeros(NbStatesWithVoid, dtype=numpy.int8)
            CN2PathProba = 0

            for currentState in range(1, NbStatesWithVoid):
                probMax = -1
                prevStateMax = -1
                for prevState in range(NbStatesWithVoid):
                    # probability of path coming from prevState to currentState
                    prob = (probsPrev[prevState] *
                            adjustedTransMatrix[prevState, currentState] *
                            likelihoods[exonIndex, currentState - 1])
                    # currentState - 1 because likelihoods don't have values for void state
                    print("prev=", prevState, ", current=", currentState, ",prob=", prob)
                    if prob > probMax:
                        probMax = prob
                        prevStateMax = prevState
                    if (currentState == 3) and ((prevState == 0) or (prevState == 3)):
                        if prob > CN2PathProba:
                            CN2PathProba = prob

                # save most likely path leading to currentState
                probsCurrent[currentState] = probMax
                bestPrevState[currentState] = prevStateMax

            # if all LIVE states (probsCurrent > 0) at currentExon have the same
            # predecessor state and that state is CN2 : backtrack from [previous exon, CN2]
            # and reset
            if (((probsCurrent[1] == 0) or (bestPrevState[1] == 3)) and
                ((probsCurrent[2] == 0) or (bestPrevState[2] == 3)) and
                ((probsCurrent[3] == 0) or (bestPrevState[3] == 3)) and
                ((probsCurrent[4] == 0) or (bestPrevState[4] == 3)) and
                (len(calledExons) > 0)):
                print('RESET')
                CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2PathProbas, 3, sampleID))

                # start new paths at CN2 in previous exon with a starting path proba of 1.
                bestPrevState[1:] = 3
                for state in range(1, NbStatesWithVoid):
                    probsCurrent[state] = (adjustedTransMatrix[3, state] *
                                           likelihoods[exonIndex, state - 1])
                # reset all buildCNVs() accumulators
                calledExons = []
                path = []
                bestPathProbas = []
                CN2PathProbas = []

            # Update previous probabilities and move to the next exon
            numpy.copyto(probsPrev, probsCurrent)
            prevEnd = exons[exonIndex][2]

            calledExons.append(exonIndex)
            path.append(bestPrevState)
            bestPathProbas.append(probsCurrent.copy())
            CN2PathProbas.append(CN2PathProba)

        # Final backtrack to aggregate calls for the last exon
        if len(calledExons) > 0:
            print("FINALRESET")
            CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2PathProbas,
                                  bestPathProbas[-1].argmax(), sampleID))

        return (CNVs)

    except Exception as e:
        logger.error("CNCalls failed for sample %s - error: %s - exonIndex %s", sampleID, repr(e), str(exonIndex))
        raise Exception(sampleID)


######################################
# buildCNVs
# Identify CNVs (= consecutive exons with the same CN) in a most-likely path, and
# calculate the associated "qualityScore" (see below).
# Requirement: the called exon preceding calledExons[0] (called the "path root") must
# be in state CN2 in every most likely path.
#
# Args:
# - calledExons [list of ints]: list of called exonIndexes to process here
# - path (list of len(calledExons) ndarrays of NbStates ints):
#   path[e][s] == state of called exon preceding calledExons[e] that produces the max
#   proba for state s at exon calledExons[e]
# - bestPathProbas (list of len(calledExons) ndarrays of NbStates floats):
#   bestPathProbas[e][s] == proba of most likely path ending in state s at exon
#   calledExons[e] and starting at the path root
# - CN2PathProbas (list of len(calledExons) floats): CN2FromCN2Probas[e] == proba of
#   path ending in state CN2 at exon calledExons[e], starting at path root, and staying
#   in state CN2 all along
# - lastState [int]: state with the max probability for the last exon in calledExons
# - sampleID [str]
#
# Returns a list of CNVs, a CNV == [CNType, startExon, endExon, qualityScore, sampleID]:
# - CNType is 0-3 (== CN)
# - startExon and endExon are indexes (in the global exons list) of the first and
#   last exons defining this CNV
# - qualityScore = proba of most likely path between the called exon immediately
#   preceding and immediately following the CNV, divided by the proba of
#   the CN2-only path between the same exons
def buildCNVs(calledExons, path, bestPathProbas, CN2PathProbas, lastState, sampleID):
    CNVs = []

    print("CalledEx:", calledExons)
    print("path:", path)
    print("bestPathProbas:", bestPathProbas)
    print("CN2PathProbas:", CN2PathProbas)
    print("lastState:", lastState)

    if lastState != 2:
        # can only happen when called with the last exon of a chrom: append bogus last
        # exon in CN2 state, copying the path proba
        calledExons.append(-1)
        path.append(numpy.array([0, 0, lastState, 0]))
        bestPathProbas.append(numpy.array([0, 0, bestPathProbas[-1][lastState], 0]))
        CN2PathProbas.append(CN2PathProbas[-1])

    # build ndarray of states that form the most likely path, must start from the end
    mostLikelyStates = numpy.zeros(len(calledExons), dtype=numpy.int8)
    mostLikelyStates[-1] = 2
    currentState = 2
    for cei in range(len(calledExons) - 1, 0, -1):
        currentState = path[cei][currentState]
        mostLikelyStates[cei - 1] = currentState

    # now walk through the path of most likely states, constructing CNVs as we go
    currentState = mostLikelyStates[0]
    firstExonInCurrentState = 0

    for cei in range(1, len(calledExons)):
        if mostLikelyStates[cei] == currentState:
            # next exon is in same state, NOOP
            continue
        else:
            if (currentState != 2):
                # we changed states and current wasn't CN2, create CNV
                # score = ratio between best path proba and CN2-only path proba
                qualityScore = bestPathProbas[cei][mostLikelyStates[cei]] / CN2PathProbas[cei]
                if firstExonInCurrentState > 0:
                    # we want the probas of the paths starting at the exon immediately
                    # preceding the CNV, not starting at the path root
                    qualityScore /= bestPathProbas[firstExonInCurrentState - 1][mostLikelyStates[firstExonInCurrentState - 1]]
                    qualityScore *= CN2PathProbas[firstExonInCurrentState - 1]
                CNVs.append([currentState, calledExons[firstExonInCurrentState],
                             calledExons[cei - 1], qualityScore, sampleID])
            # in any case we changed states, update accumulators
            currentState = mostLikelyStates[cei]
            firstExonInCurrentState = cei

    return(CNVs)
