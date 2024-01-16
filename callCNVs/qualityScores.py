import logging
import numpy as np

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
###################################
# calcQualityScore
# calculates quality scores for a list of CNVs by comparing their path probabilities
# with the baseline CN2 state probabilities.
#
# Args:
# - CNVs (list): CNV information. Each CNV is represented as [CNType[int], exonStart[int],
#    exonEnd[int], pathProb[float], sampleName[str]].
# - likelihoods (dict): key==sampleID, value==np.array[floats] likelihoods
# - transMatrix (numpy.ndarray): Transition matrix used in the HMM Viterbi algorithm.
#
# Returns:
# - qualityScores (list[floats]): Quality scores for each CNV. The score is a log-odds
#    measure comparing the CNV's path probability with the CN2 probability.
def calcQualityScore(CNVs, likelihoods, transMatrix):
    qualityScores = []
    for cnv in CNVs:

        exonIndexStart, exonIndexEnd, cnProbs, sampID = cnv[1], cnv[2], cnv[3], cnv[4]
        if cnProbs == 0:
            print(cnProbs, exonIndexStart, exonIndexEnd, cnProbs, sampID)
            continue

        score = calculateCNVQualityScore(cnProbs, likelihoods[sampID][exonIndexStart:exonIndexEnd + 1, 2], transMatrix[3][3])
        qualityScores.append(score)
    return qualityScores


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
###################################
# calculateCNVQualityScore
# calculates a quality score for a single CNV. The score is derived from the
# CNV's path probability and the CN2 state probability for the CNV's exons.
#
# Args:
# - pathProb [float]: The path probability of the CNV.
# - cnvCN2likelihoods (numpy.ndarray[floats]): CN2 likelihoods for the CNV's exons.
# - CN2toCN2probs (float): Transition probability from CN2 to CN2 state.
#
# Returns:
# - logOdds [float]: The calculated quality score (log odds) for the CNV.
def calculateCNVQualityScore(cnProbs, cnvCN2likelihoods, CN2toCN2probs):
    # Calculate the CN2 probabilities
    CN2Probs = calculateCN2Probs(cnvCN2likelihoods, CN2toCN2probs)

    # Calculate the quality score (log odds)
    logOdds = np.log10(cnProbs / CN2Probs)
    return logOdds


###################################
# calculateCN2ProbsAverage
# calculates the aggregated probability for the CN2 state across a range of exons.
# the optimal path, if there is no change and it remains at CN2, CN2 to CN2 transition
# probabilities are used, reflecting the stability of this state.
#
# Args:
# - cnvCN2likelihoods (numpy.ndarray): CN2 likelihoods for a range of exons.
# - CN2toCN2probs [float]: transition probability from CN2 to CN2 state.
#
# Returns:
# - CN2Probs [float]: aggregated probability for the CN2 state.
def calculateCN2Probs(cnvCN2likelihoods, CN2toCN2probs):
    # Extract probabilities for specified exons and exclude 'no call'
    validCN2Probs = cnvCN2likelihoods[cnvCN2likelihoods != -1]
    probsPrev = 1
    CN2Probs = 0
    for likelihoods in validCN2Probs:
        CN2Probs = probsPrev * CN2toCN2probs * likelihoods
        probsPrev = CN2Probs

    return CN2Probs
