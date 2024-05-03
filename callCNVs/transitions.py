import logging
import numpy

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# buildBaseTransMatrix
# Build the base transition matrix, using the states that maximize
# the posterior probability (ie prior * likelihood) for each
# pair of called exons that are "close enough" (maxIED).
#
# Args:
# - likelihoodsDict: key==sampleID, value==(ndarray[floats] dim NbExons*NbStates)
#   holding the likelihoods of each state for each exon for this sample (no-call
#   exons should have all likelihoods == -1)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - priors (ndarray dim nbStates): prior probabilities for each state
# - maxIED (int): max inter-exon distance for a pair of consecutive called exons
#   to be used here
#
# Returns transMatrix (ndarray[floats] dim nbStates*nbStates): base transition
# probas between states
def buildBaseTransMatrix(likelihoodsDict, exons, priors, maxIED):

    ##### TODO I AM HERE


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
