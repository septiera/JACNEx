import logging
import numpy as np

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
###################################
# getCNVsQualityScore
# This function is the main entry point for calculating quality scores for a list of CNVs.
# It iterates over each CNV in the provided list, calls calculateCNVQualityScore for each,
# and compiles the scores into a list.
# This approach allows for a comprehensive assessment of the quality of each CNV detection
# within the context of the given dataset, taking into account both the individual CNV
# characteristics and the overall sequencing data profile.
#
# Args:
# - CNVs (list[str, int, int, int, floats, str]): CNV infos [chromType, CNType, exonStart, exonEnd, pathProb, sampleName]
# - CNPath_A (dict): key=sampID, value=2D numpy arrays [floats]
#                    representing CN states probabilities for exon calls (no call==-1).
#                    dim = NbExonsOnAutosomes * NbOfCNStates
# - CNPath_G (dict): same as CNProbs_A but for gonosomes
#
# Return a list of floats, where each element is the quality score corresponding to a CNV in
# the CNVs list. This list is of the same length as CNVs.
def getCNVsQualityScore(CNVs, CNProbs):
    quality_scores = []
    for cnv in CNVs:
        score = calculateCNVQualityScore(cnv, CNProbs)
        quality_scores.append(score)
    return quality_scores


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
###################################
# calculateCNVQualityScore
# This function calculates a quality score for a given CNV.
# The score is computed by dividing the path probability of the CNV (viterbi most probable path)
# by the average CN2 probability for the range of exons affected by the CNV.
# This method provides a measure of the CNV's reliability compared to the baseline CN2
# state probabilities.
#
# Args:
# - CNVs (list[str, int, int, int, floats, str]): CNV infos [chromType, CNType, exonStart, exonEnd, pathProb, sampleName]
# - CNPath_A (dict): key=sampID, value=2D numpy arrays [floats]
#                    representing CN states probabilities for exon calls (no call==-1).
#                    dim = NbExonsOnAutosomes * NbOfCNStates
# - CNPath_G (dict): same as CNProbs_A but for gonosomes
def calculateCNVQualityScore(cnv, CNProbs):
    # Extract necessary information from the CNV
    cn, exonStart, exonEnd, pathProb, sampleName = cnv[0], cnv[1], cnv[2], cnv[3], cnv[4],

    # Calculate the average of CN2 probabilities
    CN2Average = calculateCN2ProbsAverage(CNProbs, exonStart, exonEnd, sampleName)

    logOdds = 0

    if pathProb == 0:
        print(cn, exonStart, exonEnd, pathProb, sampleName, CN2Average)
    # Check if the CN2 average is not zero
    elif CN2Average != 0:
        # Calculate the quality score (log odds)
        logOdds = np.log10(pathProb / CN2Average)

    return logOdds


###################################
# calculateCN2ProbsAverage
# This function calculates the average probability of the CN2 state for a specified
# range of exons.
# It is designed to handle sequencing data from either autosomes or gonosomes, based
# on the chromosome type provided.
# The function iterates through the probability matrices for each sample, extracting
# and averaging the CN2 probabilities for the specified exon range, excluding any 'no call' entries.
#
# Args:
# - CNProbs (dict): key=='A' for autosomes or 'G' for gonosomes, value==dictionary (key==sample ID,
#                   value==2D numpy array of CN state probabilities [floats].
# - chromType [str]: 'A' for autosomes or 'G' for gonosomes
# - exonStart [int]: index representing the starting exon.
# - exonEnd [int]: index representing the ending exon.
# - sampleName [str]: sample identifier for which the CN2 probabilities are to be calculated.
#
# Return a float representing the average CN2 probability across all samples for the
# specified exon range. Returns 0 if there are no valid CN2 probabilities.
def calculateCN2ProbsAverage(CNProbs, exonStart, exonEnd, sampleName):
    # Retrieve the probability array for the specified sample
    probs = CNProbs[sampleName]

    # Extract probabilities for specified exons and exclude 'no call'
    CN2Probs4Sample = probs[exonStart:exonEnd, 2]
    validCN2Probs = CN2Probs4Sample[CN2Probs4Sample != -1]

    # Calculate the average if the list is not empty
    if len(validCN2Probs) > 0:
        averageCN2Probs = np.mean(validCN2Probs)
    else:
        averageCN2Probs = 0

    return averageCN2Probs
