import logging
import numpy
import math

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
# Specs:
# 1. calculates the starts of chromosomes and computes the distances between consecutive exons within each chromosome.
#    This is done separately for autosomal and gonosomal exons.
# 2. calculates the maximum distance (dmax) below a specified percentile threshold and the median distance (medDist)
#    of non-zero distances.
#    These values are used to determine the threshold for considering exon distances in the transition matrix update.
# 3. initializes a transition matrix, which is a 2D array of integers representing the transition probabilities
#    between different CN states.
# 4. update transition matrix with CN counts for autosomal and gonosomal samples separately.
#    This update process is based on the likelihoods of CN states provided for each sample,
#    as well as prior probabilities and exon distances.
#    Exons with distances exceeding medDist are excluded from the transition matrix update.
# 5. normalize the transition matrix of counts to ensure that the probabilities sum to 1.
#    Additionally, an 'init' state is added to the transition matrix to initialize a HMM for further analysis.
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
# - transMatrixCutoff (int): inter-exon percentile threshold for initializing the transition matrix.
#
# Returns:
# - transAndInit (np.ndarray[floats]): transition matrix used for the HMM, including
#                                      the "init" state. dim = [nbStates+1, nbStates+1]
# - dmax (int): maximum distance below the specified percentile threshold.
def getTransMatrix(likelihoods_A, likelihoods_G, autosomeExons, gonosomeExons,
                   priors, nbStates, transMatrixCutoff):

    dmax, medDist = getMaxAndMedianDist(autosomeExons, gonosomeExons, transMatrixCutoff)
    logger.info("dmax below the %sth percentile threshold is : %s", transMatrixCutoff, dmax)
    logger.info("median inter-exon distance is : %s", medDist)

    # initialize the transition matrix
    # 2D array [ints], expected format for a transition matrix [i; j]
    # contains all prediction counts of states
    transitions = numpy.zeros((nbStates, nbStates), dtype=int)
    # update the transition matrix with CN probabilities for autosomal samples
    updateTransMatrix(transitions, likelihoods_A, priors, autosomeExons, medDist)
    # repeat the process for gonosomal samples
    updateTransMatrix(transitions, likelihoods_G, priors, autosomeExons, medDist)
    # normalize the transition matrix and add an 'init' state
    transAndInit = formatTransMatrix(transitions, priors, nbStates)

    return (transAndInit, dmax)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# getMaxAndMedianDist
# calculates two important metrics for constructing the transition matrix of a HMM:
# - dmax (maximum distance): This value serves as a threshold for adjusting transition probabilities
#                            in the Viterbi process.
# - medDist (median distance): Exons separated by a distance less than or equal to medDist are
#                              considered sufficiently close for constructing the transition matrix.
#
# Args:
# - autosomeExons, gonosomeExons (list of lists[str, int, int, str]): exons infos [CHR, START, END, EXONID].
# - transMatrixCutoff [int]: inter-exon percentile threshold for initializing the transition matrix.
#
# Returns:
# - tuple: A tuple containing the maximum distance (dmax) and the median distance (medDist).
def getMaxAndMedianDist(autosomeExons, gonosomeExons, transMatrixCutoff):
    # concatenate exons from autosomes and gonosomes into a single list
    exons = autosomeExons + gonosomeExons

    # initialize a list to store distances between consecutive exons
    distExons = []
    # initialize variables to keep track of the previous chromosome
    # and its end position
    prev_chr = ""
    prev_end = 0

    for exon in exons:
        chr_name, start, end, _ = exon
        # check if the current chromosome is different from the previous one
        if prev_chr != chr_name:
            prev_chr = chr_name
            prev_end = end
            continue
        # calculate the distance between the current exon and the previous
        # one on the same chromosome.
        # if exons are adjacent, the distance will be 0.
        dist = start - prev_end - 1
        if dist > 0:  # only positive distance
            distExons.append(dist)
        prev_end = end

    medDist = numpy.median(distExons)
    sorted_distances = sorted(distExons)  # ascending order

    # Calculate the index corresponding to the specified percentile
    index = int(len(sorted_distances) * transMatrixCutoff / 100)

    return (sorted_distances[index], medDist)


#########################################
# updateTransMatrix
# Updates the transition matrix in place based on the most probable CN states
# for each exon.
#
# Args:
# - transitions (np.ndarray[floats]): Transition matrix to be updated, dimensions [NBStates, NBStates].
# - likelihoodsDict (dict): CN likelihoods per sample, keys = sampID,
#                           values = 2D numpy arrays representing CN probabilities for each exon.
# - priors (np.ndarray): Prior probabilities for CN states.
# - exons (list of lists[str, int, int, str]): exons infos [CHR, START, END, EXONID].
# - maxDistBetweenExons (int): distance cutoff
def updateTransMatrix(transitions, likelihoodsDict, priors, exons, maxDistBetweenExons):
    for likelihoodsArr in likelihoodsDict.values():
        # determines for each sample the most probable CN state for each exon
        # by finding the maximum probability in the CN probabilities array.
        CNPath = callCNVs.priors.getCNPath(likelihoodsArr, priors)

        # initialize to bogus chrom
        prevCN = -1
        prevChrom = -1
        prevEnd = -1

        for ei in range(len(CNPath)):
            # skip no-call exons
            if CNPath[ei] == -1:
                continue

            # if new chrom first called exon is not counted but used for init
            if prevChrom != exons[ei][0]:
                prevCN = CNPath[ei]
                prevChrom = exons[ei][0]
                prevEnd = exons[ei][2]
                continue

            dist = exons[ei][1] - prevEnd - 1

            # update transitions if exons are close enough.
            if dist <= maxDistBetweenExons:
                transitions[prevCN, CNPath[ei]] += 1

            # in both cases update prevs
            prevEnd = exons[ei][2]
            prevCN = CNPath[ei]


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
