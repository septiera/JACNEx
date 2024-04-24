import logging
import numpy

####### JACNEx modules
import callCNVs.priors

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
# - priors (,p.ndarray[floats]): prior probabilities for each CN status.
# - nbStates (int): number of CN states.
# - transMatrixCutoff (int): percentile threshold for calculating the maximum distance between exons.
#
# Returns:
# - transAndInit (np.ndarray[floats]): transition matrix used for the HMM, including
#                                      the "init" state. dim = [nbStates+1, nbStates+1]
# - dmax (int): maximum distance below the specified percentile threshold.
def getTransMatrix(likelihoods_A, likelihoods_G, autosomeExons, gonosomeExons,
                   priors, nbStates, transMatrixCutoff):
    # get the chromosome starts and exon distances
    isFirstExon_A, distExons_A = getChromosomeStartsAndExonDistances(autosomeExons)
    isFirstExon_G, distExons_G = getChromosomeStartsAndExonDistances(gonosomeExons)

    dmax, medDist = getDistMaxAndMedianDist(distExons_A, distExons_G, transMatrixCutoff)
    logger.info("dmax below the %sth percentile threshold is : %s", transMatrixCutoff, dmax)
    logger.info("median distance is : %s", medDist)

    # initialize the transition matrix
    # 2D array [ints], expected format for a transition matrix [i; j]
    # contains all prediction counts of states
    transitions = numpy.zeros((nbStates, nbStates), dtype=int)
    # update the transition matrix with CN probabilities for autosomal samples
    transitions = updateTransMatrix(transitions, likelihoods_A, priors, isFirstExon_A, distExons_A, medDist)
    # repeat the process for gonosomal samples
    transitions = updateTransMatrix(transitions, likelihoods_G, priors, isFirstExon_G, distExons_G, medDist)

    # normalize the transition matrix and add an 'init' state
    transAndInit = formatTransMatrix(transitions, priors, nbStates)

    return (transAndInit, dmax)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# getChromosomeStartsAndExonDistances:
# Identifies the start of each chromosome in the list of exons and calculates the distance
# between consecutive exons within the same chromosome.
#
# Args:
# - exons (list of lists): Each inner list contains information about an exon
#                             in the format [CHR, START, END, EXON_ID].
#
# Returns:
# - isFirstExon (np.ndarray[bool]): A boolean array where each 'True' indicates the first exon of a chromosome.
# - exonDistances (np.ndarray[int]): An array containing the distance between consecutive exons within the same chromosome.
def getChromosomeStartsAndExonDistances(exons):
    # initialize a boolean array with False
    isFirstExon = numpy.zeros(len(exons), dtype=bool)
    distExons = numpy.zeros(len(exons), dtype=int)
    prev_chr = ""
    prev_end = 0

    for ei in range(len(exons)):
        chr_name, start, end, _ = exons[ei]
        # Check if the current exon belongs to a new chromosome
        if prev_chr != chr_name:
            isFirstExon[ei] = True
            prev_chr = chr_name
            prev_end = end
        else:
            interExDist = start - prev_end - 1
            # Calculate the distance between the current exon and the previous one;
            # note that any distance equal to zero implies overlapping exons.
            if interExDist > 0:
                distExons[ei] = interExDist
            prev_end = end

    return (isFirstExon, distExons)


##########################################
# getDistMax
# calculates the maximum distance below a given percentile threshold from two lists
# of distances between exons, ignoring zero distances.
#
# Args:
# - distExons_A (np.ndarray[ints]): distances between exons for chromosome A.
# - distExons_G (np.ndarray[ints]): distances between exons for chromosome G.
# - transMatrixCutoff (float): percentile threshold below which to find the maximum distance.
#
# Returns:
# - dmax (int): Maximum distance below the specified percentile threshold.
# - medDist (float): Median distance of the non-zero distances.
def getDistMaxAndMedianDist(distExons_A, distExons_G, transMatrixCutoff):
    # Concatenate the two lists of distances
    all_distances = numpy.concatenate([distExons_A, distExons_G])

    # Remove zero distances
    non_zero_distances = all_distances[all_distances != 0]

    medDist = numpy.median(non_zero_distances)

    # Sort the non-zero distances
    sorted_distances = numpy.sort(non_zero_distances)

    # Calculate the index corresponding to the specified percentile
    index = int(len(sorted_distances) * transMatrixCutoff / 100)

    return (sorted_distances[index], medDist)


#########################################
# updateTransMatrix
# Updates the transition matrix based on the most probable CN states
# for each exon.
#
# Args:
# - transitions (np.ndarray[floats]): Transition matrix to be updated, dimensions [NBStates, NBStates].
# - likelihoodsDict (dict): CN likelihoodss per sample, keys = sampID,
#                           values = 2D numpy arrays representing CN probabilities for each exon.
# - priors (np.ndarray): Prior probabilities for CN states.
# - isFirstExon (np.ndarray[bool]): each 'True' indicates the first exon of a chromosome.
# - distExons (np.ndarray[ints]): Array containing distances between exons.
# - medDist (float): Median distance of the non-zero distances.
#
# Returns:
# - transitions (np.ndarray[ints]): Updated transition matrix.
def updateTransMatrix(transitions, likelihoodsDict, priors, isFirstExon, distExons, medDist):
    for sampID, likelihoodArr in likelihoodsDict.items():
        # determines for each sample the most probable CN state for each exon
        # by finding the maximum probability in the CN probabilities array.
        CNPath = callCNVs.priors.getCNPath(likelihoodArr, priors)
        prevCN = numpy.argmax(priors)
        accumulateDist = 0
        for ei in range(len(CNPath)):
            currCN = CNPath[ei]

            # skip non-interpretable exons
            if currCN == -1:
                accumulateDist += distExons[ei]
                continue

            # skip exons with distances exceeding the specified threshold (medDist)
            # Excludes exons too far apart for meaningful transition matrix construction.
            if (distExons[ei] + accumulateDist) > medDist:
                prevCN = numpy.argmax(priors)
                accumulateDist = 0
                continue

            # reset prevCN at the start of each new chromosome
            if isFirstExon[ei]:
                prevCN = numpy.argmax(priors)
                transitions[prevCN, currCN] += 1
                continue

            # updates variables
            transitions[prevCN, currCN] += 1
            prevCN = currCN
            accumulateDist = 0

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
