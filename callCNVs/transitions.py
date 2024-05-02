import logging
import numpy

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# getTransMatrix
# Calculates the transition matrix for CN states, and computes
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

    # Normalize each row of the transition matrix
    row_sums = numpy.sum(transitions, axis=1, keepdims=True)
    normalized_arr = transitions / row_sums

    return (normalized_arr, dmax)


######################################
# adjustTransMatrix
# When distance between exons increases, the correlation between CN states of
# consecutive exons diminishes until at some point we simply expect the priors.
# Formally we implement this as follows: transition probabilities are smoothed
# from base probas to prior probas following a power law, until reaching the
# prior at d = dmax.
#
# Args:
# - transMatrix (ndarray dim 4*4): base transition probabilities between states
# - priors (ndarray dim 4): prior probabilities for each state
# - d [int]: distance between exons
# - dmax (float): distance where prior is reached
#
# Returns: adjusted transition probability matrix (ndarray dim 4*4)
def adjustTransMatrix(transMatrix, priors, d, dmax):
    # hardcoded power of the power law, values between 3-6 should be reasonable
    N = 5

    newTrans = numpy.zeros_like(transMatrix)

    if d >= dmax:
        # use priors
        for prevState in range(len(transMatrix)):
            newTrans[prevState] = priors

    else:
        # newTrans[p->i](x) = trans[p->i] + (prior[i] - trans[p->i]) * (x/d_max)^N
        for state in range(len(transMatrix)):
            newTrans[:, state] = (transMatrix[:, state] +
                                  (priors[state] - transMatrix[:, state]) * (d / dmax) ** N)

    return newTrans


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
        bestStates = (priors * likelihoodsArr).argmax(axis=1)

        # initialize to bogus chrom
        prevCN = -1
        prevChrom = -1
        prevEnd = -1

        for ei in range(len(bestStates)):
            # skip no-call exons
            if likelihoodsArr[ei][0] == -1:
                continue

            # if new chrom first called exon is not counted but used for init
            if prevChrom != exons[ei][0]:
                prevCN = bestStates[ei]
                prevChrom = exons[ei][0]
                prevEnd = exons[ei][2]
                continue

            dist = exons[ei][1] - prevEnd - 1

            # update transitions if exons are close enough.
            if dist <= maxDistBetweenExons:
                transitions[prevCN, bestStates[ei]] += 1

            # in both cases update prevs
            prevEnd = exons[ei][2]
            prevCN = bestStates[ei]
