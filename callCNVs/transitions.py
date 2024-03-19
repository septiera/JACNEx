import logging
import numpy

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# getTransMatrix
# Calculates the transition matrix for CN states including an 'init' state, and computes
# the CN probabilities for autosomal and gonosomal samples.
# Methodology:
# 1. Identifying the start of new chromosomes for both autosomal and gonosomal exons.
# 2. Computing CN probabilities for autosomal and gonosomal samples separately.
# 3. Updating a transition matrix based on the computed CN probabilities.
# 4. Normalizing the transition matrix and adding an 'init' state for Hidden Markov Model(HMM)
#    initialization.
# The computed CN probabilities serve for CNVs quality score calculations in downstream analysis.
#
# Args:
# - likelihoods_A (dict): CN likelihoods for autosomal exons; key=sampID[str],
#                         values=numpy.ndarray of likelihoods[floats].
#                         dim = NbExonsOnAutosomes * NbOfCNStates
# - likelihoods_G (dict): CN likelihoods for gonosomal exons.
# - autosomeExons (list of lists[str, int, int, str]): autosome exon infos;
#                                                      [CHR,START,END,EXONID]
# - gonosomeExons (list of lists[str, int, int, str]): gonosome exon infos
# - priors (numpy.ndarray[floats]): prior probabilities for each CN status.
# - nbStates [int]: number of CN states.
#
# Returns:
# - transAndInit (numpy.ndarray[floats]): transition matrix used for the HMM, including
#                                      the "init" state. dim = [nbStates+1, nbStates+1]
def getTransMatrix(likelihoods_A, likelihoods_G, autosomeExons, gonosomeExons,
                   priors, nbStates):

    # identify the start of new chromosomes for autosomal and gonosomal exons
    isFirstExon_A = flagChromStarts(autosomeExons)
    isFirstExon_G = flagChromStarts(gonosomeExons)

    # compute CN probabilities for autosomal and gonosomal samples
    CNProbs_A = getCNProbs(likelihoods_A, priors)
    CNProbs_G = getCNProbs(likelihoods_G, priors)

    # initialize the transition matrix
    # 2D array [ints], expected format for a transition matrix [i; j]
    # contains all prediction counts of states
    transitions = numpy.zeros((nbStates, nbStates), dtype=int)
    # update the transition matrix with CN probabilities for autosomal samples
    transitions = updateTransMatrix(transitions, CNProbs_A, priors, isFirstExon_A)
    # repeat the process for gonosomal samples
    transitions = updateTransMatrix(transitions, CNProbs_G, priors, isFirstExon_G)

    # normalize the transition matrix and add an 'init' state
    transAndInit = formatTransMatrix(transitions, priors, nbStates)

    return transAndInit


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# flagChromStarts:
# Creates a boolean vector indicating the start of a new chromosome in the exons list.
# Each element in the vector corresponds to an exon; the value is True if the exon
# is the first exon of a new chromosome, and False otherwise.
#
# Args:
# - exons (list of lists): Each inner list contains information about an exon
#                             in the format [CHR, START, END, EXON_ID].
#
# Returns:
# - isFirstExon (numpy.ndarray[bool]): each 'True' indicates the first exon of a chromosome.
def flagChromStarts(exons):
    # initialize a boolean array with False
    isFirstExon = numpy.zeros(len(exons), dtype=bool)
    prevChr = None

    for index, exon in enumerate(exons):
        chromosome = exon[0]
        if chromosome != prevChr:
            isFirstExon[index] = True
            prevChr = chromosome

    return isFirstExon


#########################################
# getCNProbs
# Computes the probabilities for all CN states for each exon in each sample.
# Methodology:
# 1. For each sample, it calculates the probabilities (odds) for each CN state
#    of each exon by multiplying the likelihood data with prior probabilities.
# 2. If an exon is marked as 'no call' (indicated by -1 in the likelihood data),
#    the function sets the probabilities for all CN states of that exon to -1.
#
# Args:
# - likelihoodDict (dict): keys=sampID, values=numpy.ndarray(NBExons,NBStates)
# - priors (numpy.ndarray): Prior probabilities for CN states.
#
# Returns:
# - CNProbs (dict): CN probabilities per sample, keys = sampID, values = 2D numpy arrays
#                   representing CN probabilities for each exon.
def getCNProbs(likelihoodDict, priors):
    CNProbs = {}

    for sampID, likelihoods in likelihoodDict.items():
        odds = likelihoods * priors
        skip_exon = numpy.any(likelihoods == -1, axis=1)
        odds[skip_exon] = -1
        CNProbs[sampID] = odds

    return CNProbs


#########################################
# updateTransMatrix
# Updates the transition matrix based on the most probable CN states
# for each exon.
# Preparing the transition matrix for use in HMM.
# Methodology:
# 1. determines for each sample the most probable CN state for each exon
#    by finding the maximum probability in the CN probabilities array.
# 2. The function identifies exons with non-interpretable data ('no call') and skips these
#    exons in the update process. 'No call' exons are marked by -1 in the probabilities array.
# 3. The function resets the previous CN state (prevCN) at the start of each new chromosome.
# 4. For each exon (excluding 'no call' exons), the function updates the transition matrix.
#    It increments the count in the matrix cell corresponding to the transition from the
#    previous CN state to the current CN state.
#
# Args:
# - transitions (numpy.ndarray[floats]): Transition matrix to be updated, dimensions [NBStates, NBStates].
# - CNProbs (dict): CN probabilities per sample, keys = sampID,
#                   values = 2D numpy arrays representing CN probabilities for each exon.
# - priors (numpy.ndarray): Prior probabilities for CN states.
# - isFirstExon (numpy.ndarray[bool]): each 'True' indicates the first exon of a chromosome.
#
# Returns:
# - transitions (numpy.ndarray[ints]): Updated transition matrix.
def updateTransMatrix(transitions, CNProbs, priors, isFirstExon):
    for sampID, odds in CNProbs.items():
        # determine the most probable CN state for each exon
        CNsList = numpy.argmax(odds, axis=1)
        # identify exons with non-interpretable data
        # numpy.ndarray of boolean (0:call, 1:no call) of length NbExons
        isSkipped = numpy.any(odds == -1, axis=1)
        # start with the most probable CN state based on priors (CN2)
        prevCN = numpy.argmax(priors)

        for ei in range(len(CNsList)):
            currCN = CNsList[ei]

            # skip non-interpretable exons
            if isSkipped[ei]:
                continue

            # reset prevCN at the start of each new chromosome
            if isFirstExon[ei]:
                prevCN = numpy.argmax(priors)

            # updates variables
            transitions[prevCN, currCN] += 1
            prevCN = currCN

    return transitions


###########################################
# formatTransMatrix
# Formats the transition matrix by normalizing it and adding an 'init' state.
# Normalization ensures that each row sums to one, and the 'init' state is used for initializing
# the HMM.
#
# Args:
# - transitions (numpy.ndarray): The transition matrix to be formatted, dimensions [nbStates, nbStates].
# - priors (numpy.ndarray): Prior probabilities for CN states, used for the 'init' state.
# - nbStates (int): Number of CN states.
#
# Returns:
# - numpy.ndarray: The formatted transition matrix with normalized values and an 'init' state,
#               dimensions [nbStates+1, nbStates+1].
def formatTransMatrix(transitions, priors, nbStates):
    # Normalize each row of the transition matrix
    row_sums = numpy.sum(transitions, axis=1, keepdims=True)
    normalized_arr = transitions / row_sums

    # Add an 'init' state to the matrix
    transAndInit = numpy.vstack((priors, normalized_arr))
    transAndInit = numpy.hstack((numpy.zeros((nbStates + 1, 1)), transAndInit))

    return transAndInit
