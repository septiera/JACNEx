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
# callAllCNVs
# Call CNVs with callCNVsOneSample() in parallel for each sample in likelihoodsDict.
#
# Args:
# - likelihoodsDict: key==sampleID, value==(ndarray[floats] dim NbExons*NbStates) holding the
#   pseudo-emission probabilities (likelihoods) of each state for each exon for this sample.
# - exons: list of NbExons exons, one exon is a list [CHR, START, END, EXONID].
# - transMatrix (ndarray[floats] dim NbStates*NbStates): base transition probas between states
# - priors (ndarray dim NbStates): prior probabilities for each state
# - dmax [int]: param for adjustTransMatrix()
# - jobs (int): Number of jobs to run in parallel.
#
# Returns a list of CNVs, a CNV is a list (types [int, int, int, float, str]):
# [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleID]
# where firstExonIndex and lastExonIndex are indexes in the provided exons list.
def callAllCNVs(likelihoodsDict, exons, transMatrix, priors, dmax, jobs):
    CNVs = []

    ##################
    # Define nested callback for processing callCNVsOneSample() result (so CNVs is in scope).
    # sampleDone:
    # arg: a Future object returned by ProcessPoolExecutor.submit(callCNVsOneSample).
    # If something went wrong, log and propagate exception, otherwise store CNVs.
    def sampleDone(futureRes):
        e = futureRes.exception()
        if e is not None:
            logger.error("callCNVsOneSample() failed for sample %s", str(e))
            raise(e)
        else:
            CNVs.extend(futureRes.result())

    ##################
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        for sampID in likelihoodsDict.keys():
            futureRes = pool.submit(callCNVsOneSample, likelihoodsDict[sampID], sampID, exons, transMatrix, priors, dmax)
            futureRes.add_done_callback(sampleDone)

    return(CNVs)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

######################################
# callCNVsOneSample:
# call and return CNVs for a single sample.
# This function implements the Viterbi algorithm to find the most likely sequence of
# states (copy-number states) given the observations (likelihoods).
#
# The underlying HMM is defined by:
# - one state per copy number (0==homodel, 1==heterodel, 2==WT, 3==CN3+==DUP)
# - emission likelihoods of the sample's FPM in each state and for each exon, which have
#   been pre-calculated;
# - transition probabilities that depend on the distance to the next exon - they begin when
#   distance=0 at the "base" values defined in transMatrix, and are smoothly adjusted following
#   a power law until they reach the prior probabilities at dist dmax
#
# Args:
# - likelihoods (ndarray[floats] dim NbExons*NbStates): pseudo-emission probabilities
#   (likelihoods) of each state for each exon for one sample.
# - sampleID [str]
# - exons [list of lists[str, int, int, str]]: exon infos [chr, START, END, EXONID].
# - transMatrix (ndarray[floats] dim NbStates*NbStates): base transition probas between states
# - priors (ndarray dim NbStates): prior probabilities for each state
# - dmax [int]: param for adjustTransMatrix()
#
# Returns:
# - CNVs (list of list[int, int, int, float, str]): list of called CNVs,
#   a CNV is a list [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleID].
def callCNVsOneSample(likelihoods, sampleID, exons, transMatrix, priors, dmax):
    try:
        CNVs = []
        NbStates = len(transMatrix)
        # sanity
        if len(exons) != likelihoods.shape[0]:
            logger.error("Numbers of exons and of rows in likelihoods inconsistent")
            raise Exception(sampleID)
        if NbStates != likelihoods.shape[1]:
            logger.error("NbStates in transMatrix and in likelihoods inconsistent")
            raise Exception(sampleID)

        # Step 1: Initialize variables
        # probsPrev[s]: probability of the most likely path ending in state s at previous exon,
        # initialize path root at CN2
        probsPrev = numpy.zeros(NbStates, dtype=numpy.float128)
        probsPrev[2] = 1
        # chrom and end of previous exon - init at -dmax so first exon uses the priors
        prevChrom = exons[0][0]
        prevEnd = -dmax

        # temp data structures used by buildCNVs() and reset whenever it is called,
        # see buildCNVs() spec for info
        (calledExons, path, bestPathProbas, CN2FromCN2Probas) = ([], [], [], [])

        # Step 2: viterbi forward algorithm
        for exonIndex in range(len(exons)):
            if likelihoods[exonIndex, 0] == -1:
                # exon is no-call => skip
                continue

            if exons[exonIndex][0] != prevChrom:
                # changing chroms:
                appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas)
                CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleID))
                # reinit with CN2 as path root
                probsPrev[:] = 0
                probsPrev[2] = 1
                prevChrom = exons[exonIndex][0]
                prevEnd = -dmax
                (calledExons, path, bestPathProbas, CN2FromCN2Probas) = ([], [], [], [])

            # accumulators with current exon data for populating the buildCNVs() structures
            # bestPrevState defaults to CN2
            bestPrevState = numpy.full(NbStates, 2, dtype=numpy.int8)
            # probsCurrent[s]: probability of the most likely path ending in state s at the current exon
            probsCurrent = numpy.zeros(NbStates, dtype=numpy.float128)

            # adjust transition probabilities
            distFromPrevEx = exons[exonIndex][1] - prevEnd - 1
            adjustedTransMatrix = callCNVs.transitions.adjustTransMatrix(transMatrix, priors, distFromPrevEx, dmax)

            # calculate proba of the most likely paths ending in each state for current exon
            for currentState in range(NbStates):
                probMax = 0
                prevStateMax = 2
                for prevState in range(NbStates):
                    # probability of path coming from prevState to currentState
                    prob = (probsPrev[prevState] *
                            adjustedTransMatrix[prevState, currentState] *
                            likelihoods[exonIndex, currentState])
                    if prob > probMax:
                        probMax = prob
                        prevStateMax = prevState
                probsCurrent[currentState] = probMax
                bestPrevState[currentState] = prevStateMax

            # if all best paths for current exon have zero probability (ie they all underflowed)
            if not probsCurrent.any():
                logger.warning("in callCNVsOneSample(%s), all best paths to exon %i underflowed to zero proba. " +
                               "This should be very rare, if not please report it.", sampleID, exonIndex)
                # we'll try to build CNVs from whichever state was most likely in the last exon
                appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas)
                # make sure we will backtrack-and-reset (should already be all-2s, but whatever)
                bestPrevState[:] = 2

            # if all states at currentExon have the same predecessor state and that state is CN2:
            # backtrack from [previous exon, CN2] and reset
            if numpy.all(bestPrevState == 2):
                CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleID))
                # reinit with CN2 as path root in prev exon
                probsCurrent[:] = adjustedTransMatrix[2, :] * likelihoods[exonIndex, :]
                # bestPrevState is already all-2's
                (calledExons, path, bestPathProbas, CN2FromCN2Probas) = ([], [], [], [])

            # OK, update all structures and move to next exon
            numpy.copyto(probsPrev, probsCurrent)
            prevEnd = exons[exonIndex][2]
            calledExons.append(exonIndex)
            path.append(bestPrevState)
            bestPathProbas.append(probsCurrent)
            CN2FromCN2Probas.append(adjustedTransMatrix[2, 2] * likelihoods[exonIndex, 2])

        # Final CNVs for the last exons
        appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas)
        CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleID))

        return(CNVs)

    except Exception as e:
        logger.error("callCNVsOneSample failed for sample %s in exon %i: %s", sampleID, exonIndex, repr(e))
        raise Exception(sampleID)


######################################
# appendBogusExon:
# append a bogus exon if CN2 isn't the last calledExon's most likely state.
# This bogus exon is built so that the best path ending in its CN2 state is
# almost identical to the best path ending in calledExon[-1]'s most likely state
# (ie same path except for the appended CN2, same bestPathProbas, same score).
def appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas):
    lastState = bestPathProbas[-1].argmax()
    if lastState != 2:
        calledExons.append(-1)
        path.append(numpy.array([0, 0, lastState, 0], dtype=numpy.int8))
        bestPathProbas.append(numpy.array([0, 0, bestPathProbas[-1][lastState], 0], dtype=numpy.float128))
        CN2FromCN2Probas.append(1.0)


######################################
# buildCNVs
# Identify CNVs (= consecutive exons with the same CN) in the most-likely path ending in
# state CN2 in the last calledExon, and calculate the associated "qualityScore" (see below).
# Requirements: the called exon preceding calledExons[0] (called the "path root") must be in 
# state CN2 in the most likely path
# NOTE: the best-path-ends-in-CN2 precondition means that in the NEXT exon, all best paths 
# came from CN2 in calledExons[-1]; but it's perfectly possible that path[-1].argmax() != 2
#
# Args:
# - calledExons [list of ints]: list of called exonIndexes to process here
# - path (list of len(calledExons) ndarrays of NbStates ints):
#   path[e][s] == state of called exon preceding calledExons[e] that produces the max
#   proba for state s at exon calledExons[e]
# - bestPathProbas (list of len(calledExons) ndarrays of NbStates floats):
#   bestPathProbas[e][s] == proba of most likely path ending in state s at exon
#   calledExons[e] and starting at the path root
# - CN2FromCN2Probas (list of len(calledExons) floats): CN2FromCN2Probas[e] == proba of
#   the transition to CN2 in calledExons[e] from CN2 in previous exon
# - sampleID [str]
#
# Returns a list of CNVs, a CNV == [CNType, startExon, endExon, qualityScore, sampleID]:
# - CNType is 0-3 (== CN)
# - startExon and endExon are indexes (in the global exons list) of the first and
#   last exons defining this CNV
# - qualityScore = log10 of ratio between the proba of most likely path between the called
#   exons immediately preceding and immediately following the CNV, and the proba of
#   the CN2-only path between the same exons, capped at maxQualityScore
def buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleID):
    # max quality score produce, hard-coded here
    maxQualityScore = 100
    CNVs = []

    # if each exon's bestPath-to-CN2 comes from CN2 in prev exon,
    # the best path is necessarily all-CN2 => there's nothing to build
    # (NOTE this test is also true if calledExons is empty)
    if all((p[2] == 2) for p in path):
        return(CNVs)

    # build ndarray of states that form the most likely path, must start from the end
    mostLikelyStates = numpy.zeros(len(calledExons), dtype=numpy.int8)
    mostLikelyStates[-1] = 2
    currentState = 2
    for cei in range(len(calledExons) - 1, 0, -1):
        currentState = path[cei][currentState]
        mostLikelyStates[cei - 1] = currentState

    # sanity: path root == CN2
    if path[0][currentState] != 2:
        logger.error("in buildCNVs(), sanity check failed: path root != CN2")
        raise Exception(sampleID)

    # now walk through the path of most likely states, constructing CNVs as we go
    firstExonInCurrentState = 0
    currentState = mostLikelyStates[0]
    CN2PathProba = CN2FromCN2Probas[0]

    for cei in range(1, len(calledExons)):
        if mostLikelyStates[cei] == currentState:
            # next exon is in same state, just update CN2PathProba
            CN2PathProba *= CN2FromCN2Probas[cei]
        else:
            if (currentState != 2):
                # we are changing states and current wasn't CN2, create CNV
                # score = log10 of ratio between best path proba and CN2-only path proba,
                # use maxQualityScore if CN2-only path proba underflows to zero
                qualityScore = maxQualityScore
                CN2PathProba *= CN2FromCN2Probas[cei]
                if (CN2PathProba > 0):
                    qualityScore = bestPathProbas[cei][mostLikelyStates[cei]] / CN2PathProba
                    if firstExonInCurrentState > 0:
                        # we want the proba of the path starting at the exon immediately
                        # preceding the CNV, not starting at the path root
                        qualityScore /= bestPathProbas[firstExonInCurrentState - 1][mostLikelyStates[firstExonInCurrentState - 1]]
                    qualityScore = math.log10(qualityScore)
                    qualityScore = min(qualityScore, maxQualityScore)

                CNVs.append([currentState, calledExons[firstExonInCurrentState],
                             calledExons[cei - 1], qualityScore, sampleID])
            # in any case we changed states, update accumulators
            firstExonInCurrentState = cei
            currentState = mostLikelyStates[cei]
            CN2PathProba = CN2FromCN2Probas[cei]

    if len(CNVs) > 0:
        print("Sample=%s, calledExons=%s, path=%s, bestPathProbas=%s, CN2FromCN2Probas=%s" %
              (sampleID, str(calledExons), " ".join(map(numpy.array2string, path)),
               " ".join(map(numpy.array2string, bestPathProbas)), str(CN2FromCN2Probas)))
        print("Produced CNVs: %s" % str(CNVs))

    return(CNVs)
