import logging
import os

import numpy as np

# modules for hierachical clustering
import scipy.cluster.hierarchy
# module for PCA
import sklearn.decomposition

import figures.plots

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################
# buildClusters:
# find subsets of "similar" samples. Samples within a cluster will be used
# to fit the CN2 (ie wildtype) distribution, which in turn will allow
# to calculate likelihoods of each exon-level CN state.
#
# Args:
# - FPMarray (np.ndarray[float]): normalised fragment counts for exons on the
#   chromosome type indicated by chromType, for all samples
# - chromType (string): one of 'A', 'G' indicating that FPMarray holds counts
#   for exons on autosomes or gonosomes
# - samples: list of sampleIDs, same order as the columns of FPMarray
# - maxDist: max distance up to which we can build clusters / populate FIT_WITH
# - minSamps: min number of samples (in a cluster + its FIT_WITH friends) to declare
#   the cluster VALID
# - plotFile (str): filename (including path) for saving the dendrogram representing
#   the resulting hierarchical clustering, use "" if you don't want a plot
#
# Returns (clust2samps, fitWith, clustIsValid, linkageMatrix):
# (clust2samps, fitWith, clustIsValid) are as defined in clustFile.py parseClustsFile(), ie
# clusterIDs are formatted as TYPE_NUMBER where TYPE is 'A' or 'G', and:
# - clust2samps: dict, key==clusterID, value == list of sampleIDs
# - fitWith: dict, key==clusterID, value == list of clusterIDs
# - clustIsValid: dict, key==clusterID, value == Boolean
# - linkageMatrix (provided only for testing and debugging) is the linkage matrix
#   encoding the hierarchical clustering as returned by scipy.cluster.hierarchy.linkage,
#   ie each row corresponds to one merging step of 2 cluster (c1, c2) and holds
#   [c1, c2, dist(c1,c2), size(c1) + size(c2)]
def buildClusters(FPMarray, chromType, samples, maxDist, minSamps, plotFile):
    # reduce dimensionality with PCA
    # choosing the optimal number of dimensions is difficult, but should't matter
    # much as long as we're in the right ballpark -> our heuristic:
    # only keep dimensions that explain > minExplainedVar variance ratio
    minExplainedVar = 0.015

    # we want at most maxDims dimensions (should be more than enough for any dataset)
    maxDims = 20
    maxDims = min(maxDims, FPMarray.shape[0] - 1, FPMarray.shape[1] - 1)
    pca = sklearn.decomposition.PCA(n_components=maxDims, svd_solver='full').fit(FPMarray.T)
    logMess = "in buildClusters while choosing the PCA dimension, explained variance ratio:"
    logger.debug(logMess)
    logMess = np.array_str(pca.explained_variance_ratio_, max_line_width=200, precision=4)
    logger.debug(logMess)

    dim = 0
    while (pca.explained_variance_ratio_[dim] > minExplainedVar):
        dim += 1
    if dim == 0:
        logger.warning("in buildClusters: no informative PCA dimensions, very unusual... arbitrarily setting dim=10")
        dim = 10
    logMess = "in buildClusters, chosen number of PCA dimensions = " + str(dim)
    logger.info(logMess)
    # now fit again with only dim dimensions, and project samples
    samplesInPCAspace = sklearn.decomposition.PCA(n_components=dim, svd_solver='full').fit_transform(FPMarray.T)

    # hierarchical clustering of the samples projected in the PCA space:
    # - use 'average' method to define the distance between clusters (== UPGMA),
    #   not sure why but AS did a lot of testing and says it is the best choice
    #   when there are different-sized groups;
    # - use 'euclidean' metric to define initial distances between samples;
    # - reorder the linkage matrix so the distance between successive leaves is minimal
    linkageMatrix = scipy.cluster.hierarchy.linkage(samplesInPCAspace, method='average',
                                                    metric='euclidean', optimal_ordering=True)

    # build clusters from the linkage matrix
    (clust2samps, fitWith, clustIsValid) = linkage2clusters(linkageMatrix, chromType, samples,
                                                            maxDist, minSamps)

    # sort samples in clust2samps and clusters in fitWith (for cosmetics)
    for clust in clust2samps:
        clust2samps[clust].sort()
        fitWith[clust].sort()

    # produce and plot dendrogram
    if (plotFile != "") and os.path.isfile(plotFile):
        logger.warning("buildClusters can't produce a dendrogram: plotFile %s already exists",
                       plotFile)
    elif (plotFile != ""):
        title = "chromType=" + chromType + "  dim=" + str(dim) + "  maxDist=" + str(maxDist)
        figures.plots.plotDendrogram(linkageMatrix, samples, clust2samps, fitWith, clustIsValid,
                                     title, plotFile)

    return(clust2samps, fitWith, clustIsValid, linkageMatrix)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################


#############################
# Build clusters from the linkage matrix.
#
# Args:
# - linkageMatrix: as returned by scipy.cluster.hierarchy.linkage
# - chromType, samples: same args as buildClusters(), used for formatting/populating
#   the returned data structures
#
# Returns (clust2samps, fitWith) as specified in the header of buildClusters()
def linkage2clusters(linkageMatrix, chromType, samples):
    numNodes = linkageMatrix.shape[0]
    numSamples = numNodes + 1

    # max depth for calculating branch-length zscores
    maxDepth = 10
    BLzscores = calcBLzscores(linkageMatrix, maxDepth)

    # merge heuristic: a child wants to merge with its brother if it doesn't have any
    # fitWith clusters (otherwise the dendrogram can show non-contiguous clusters, this
    # doesn't make sense), and:
    # - parent's distance (ie height) is <= startDist, or
    # - parent isn't "too far" relative to the child's intra-cluster branch lengths,
    #   ie BLzscore <= maxZscoreToMerge
    # current startDist heurisitic: 10% of highest node
    startDist = linkageMatrix[-1][2] * 0.1
    maxZscoreToMerge = 3
    logger.debug("in buildClusters, using startDist = %.2f and maxZscoreToMerge = %f", startDist, maxZscoreToMerge)

    ################
    # a (potential) cluster, identified by its clusterIndex i (0 <= i < numSamples + numNodes), is:
    # - a list of sample indexes, stored in clustSamples[i]
    # - a list of cluster indexes to use as fitWith, in clustFitWith[i]
    clustSamples = [None] * (numSamples + numNodes)
    clustFitWith = [None] * (numSamples + numNodes)

    # When examining internal node i:
    # - if both children want to merge: they are both cleared and node i is created
    # - if only child c1 wants to merge: c1 will use c2 + clustFitWith[c2] as fitWith,
    #   c2 is left as-is, node i is created with no samples and full fitWith
    # - if no child wants to merge: they both stay as-is, and node i is created with
    #   no samples and full fitWith
    # At the end, the only clusters to create are those with clustSamples != None.

    ################
    # leaves, ie singleton samples
    for i in range(numSamples):
        clustSamples[i] = [i]
        clustFitWith[i] = []

    # internal nodes
    for ni in range(numNodes):
        (c1, c2, dist, size) = linkageMatrix[ni]
        c1 = int(c1)
        c2 = int(c2)
        children = [c1, c2]
        thisClust = ni + numSamples
        # wantsToMerge: list of 2 Bools, one for each child in the order (c1,c2)
        wantsToMerge = [False, False]
        for ci in range(2):
            if dist <= startDist:
                wantsToMerge[ci] = True
            elif (not clustFitWith[children[ci]]) and (BLzscores[ni][ci] <= maxZscoreToMerge):
                wantsToMerge[ci] = True
            # else ci doesn't want to merge with his brother

        if wantsToMerge[0] and wantsToMerge[1]:
            clustSamples[thisClust] = clustSamples[c1] + clustSamples[c2]
            clustFitWith[thisClust] = []
            # clear nodes c1 and c2
            clustSamples[c1] = None
            clustSamples[c2] = None
        else:
            # at most one child wants to merge
            # in all cases thisClust will be a virtual no-sample cluster (will be useful
            # later if thisClust is used as fitWith for another cluster)
            clustSamples[thisClust] = None
            clustFitWith[thisClust] = clustFitWith[c1] + clustFitWith[c2]
            if clustSamples[c1]:
                clustFitWith[thisClust].append(c1)
            if clustSamples[c2]:
                clustFitWith[thisClust].append(c2)

            if wantsToMerge[0]:
                # c1 wants to merge but not c2
                clustFitWith[c1] += clustFitWith[c2]
                if clustSamples[c2]:
                    # c2 is a real cluster with samples, not a virtual "fitWith" cluster
                    clustFitWith[c1].append(c2)
            elif wantsToMerge[1]:
                clustFitWith[c2] += clustFitWith[c1]
                if clustSamples[c1]:
                    clustFitWith[c2].append(c1)
            # else c1 and c2 don't change

            # DEBUGGING CODE
            if clustFitWith[thisClust]:
                logger.debug("clust %i, non-empty fitWith: %s", thisClust,
                             str(clustFitWith[thisClust]))
                logger.debug("children are %i (fitWith = %s) and %i (fitWith = %s)",
                             c1, clustFitWith[c1], c2, clustFitWith[c2])

    ################
    # populate the clustering data structures from the Tmp lists, with proper formatting
    clust2samps = {}
    fitWith = {}

    # populate clustIndex2ID with correctly constructed clusterIDs for each non-virtual cluster index
    clustIndex2ID = [None] * len(clustSamples)

    # we want clusters to be numbered increasingly by decreasing numbers of samples
    clustIndexes = []
    # find all non-virtual cluster indexes
    for thisClust in range(len(clustSamples)):
        if clustSamples[thisClust]:
            clustIndexes.append(thisClust)
    # sort them by decreasing numbers of samples
    clustIndexes.sort(key=lambda ci: len(clustSamples[ci]), reverse=True)

    nextClustNb = 1
    for thisClust in clustIndexes:
        # left-pad with leading zeroes if less than 2 digits (for pretty-sorting, won't
        # sort correctly if more than 100 clusters but it's just cosmetic)
        clustID = chromType + '_' + f"{nextClustNb:02}"
        clustIndex2ID[thisClust] = clustID
        nextClustNb += 1
        clust2samps[clustID] = [samples[i] for i in clustSamples[thisClust]]
        clust2samps[clustID].sort()
        fitWith[clustID] = []

    for thisClust in clustIndexes:
        if clustFitWith[thisClust]:
            clustID = clustIndex2ID[thisClust]
            fitWith[clustID] = [clustIndex2ID[i] for i in clustFitWith[thisClust]]
            fitWith[clustID].sort()

    return (clust2samps, fitWith)


#############################
# Given a hierarchical clustering (linkage matrix), calculate branch-length pseudo-Zscores:
# for each internal node ni (row index in linkageMatrix), BLzscores[ni] is a list of 2
# floats, one for each child cj (j==0 or 1, same order as they appear in linkageMatrix):
# BLzscores[ni][j] = [d(ni, cj) - mean(BLs[cj])] / (stddev(BLs[cj]) + 1)
# [convention: BLzscores[clusti][cj] == 0 if cj is a leaf]
# Note: divisor is stddev+1 to avoid divByZero and/or inflated zscores when BLs are equal
# or very close.
#
# Args:
# - linkageMatrix: as returned by scipy.cluster.hierarchy.linkage
# - maxDepth: max depth of branches below each child cj that are taken into account to
#   calculate mean and stddev
#
# Returns BLzscores, a list (size == number of rows in linkageMatrix) of lists of 2 floats,
# as defined above.
def calcBLzscores(linkageMatrix, maxDepth):
    numNodes = linkageMatrix.shape[0]
    numSamples = numNodes + 1
    BLzscores = [None] * numNodes
    # for each internal node index ni, BLs[ni] will be a list (index D < maxDepth) of
    # lists of floats: the lengths of branches between depths D and D+1 below the node.
    BLs = [None] * numNodes
    for ni in range(numNodes):
        (c1, c2, dist, size) = linkageMatrix[ni]
        c1 = int(c1)
        c2 = int(c2)
        BLzscores[ni] = []
        BLs[ni] = [None] * maxDepth
        for d in range(maxDepth):
            BLs[ni][d] = []
        for child in (c1, c2):
            if child < numSamples:
                # leaf
                BLzscores[ni].append(0)
                BLs[ni][0].append(dist)
            else:
                child -= numSamples
                thisBL = dist - linkageMatrix[child][2]
                childBLs = np.concatenate(BLs[child])
                thisZscore = (thisBL - np.mean(childBLs)) / (np.std(childBLs) + 1)
                BLzscores[ni].append(thisZscore)
                BLs[ni][0].append(thisBL)
                for depth in range(1, maxDepth):
                    if len(BLs[child]) >= depth:
                        BLs[ni][depth].extend(BLs[child][depth - 1])
    return(BLzscores)
