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
# - maxDist, minSamps: same args as buildClusters(), used for constructing the clusters
#
# Returns (clust2samps, fitWith, clustIsValid) as specified in the header of buildClusters()
def linkage2clusters(linkageMatrix, chromType, samples, maxDist, minSamps):
    numSamples = len(samples)
    ################
    # step 1:
    # - populate clustersTmp, a list of lists of ints:
    #   clustersTmp[i] is the list of sample indexes that belong to cluster i (at this step),
    #   this will be cleared at a later step if i is merged with another cluster
    # NOTE preallocate so we can use clustersTmp[i] = "toto"
    clustersTmp = [None] * (numSamples + linkageMatrix.shape[0])
    # - also populate fitWithTmp, a list of lists of ints, fitWithTmp[i] is the list
    #   of (self-sufficient) cluster indexes that can be used for fitting CN2 in cluster i
    fitWithTmp = [None] * (numSamples + linkageMatrix.shape[0])
    # - also populate clustSizeTmp, clustSizeTmp[i] is the total number of samples in
    #   cluster i PLUS all clusters listed in fitWithTmp[i]
    clustSizeTmp = [0] * (numSamples + linkageMatrix.shape[0])

    # initialize structures: first numSamples "clusters" are singletons with just the
    # corresponding sample
    for sample in range(numSamples):
        clustersTmp[sample] = [sample]
        clustSizeTmp[sample] = 1
        fitWithTmp[sample] = []

    # parse linkageMatrix
    for thisClust in range(linkageMatrix.shape[0]):
        (c1, c2, dist, size) = linkageMatrix[thisClust]
        c1 = int(c1)
        c2 = int(c2)
        size = int(size)
        thisClust += numSamples

        if dist <= startDist:
            clustersTmp[thisClust] = clustersTmp[c1] + clustersTmp[c2]
            clustSizeTmp[thisClust] = size
            fitWithTmp[thisClust] = []
            for childClust in (c1, c2):
                clustersTmp[childClust] = None
                clustSizeTmp[childClust] = 0
                fitWithTmp[childClust] = None

        elif dist <= maxDist:
            # between startDist and maxDist: only merge / add to fitWithTmp if a cluster
            # needs more friends, ie it is too small (counting the fitWith samples)
            if (clustSizeTmp[c1] < minSamps) and (clustSizeTmp[c2] < minSamps):
                # c1 and c2 both still needs friends => merge them
                clustersTmp[thisClust] = clustersTmp[c1] + clustersTmp[c2]
                clustSizeTmp[thisClust] = size
                fitWithTmp[thisClust] = fitWithTmp[c1] + fitWithTmp[c2]
                for childClust in (c1, c2):
                    clustersTmp[childClust] = None
                    clustSizeTmp[childClust] = 0
                    fitWithTmp[childClust] = None
            else:
                # at least one of (c1,c2) is self-sufficient, find which one it is and switch
                # them if needed so that c2 is always self-sufficient
                if (clustSizeTmp[c2] < minSamps):
                    # c2 not self-sufficient, switch with c1
                    cTmp = c1
                    c1 = c2
                    c2 = cTmp
                if (clustSizeTmp[c1] < minSamps):
                    # c2 is (now) self-sufficient but c1 isn't =>
                    # don't touch c2 but use it for fitting thisClust == ex-c1
                    clustersTmp[thisClust] = clustersTmp[c1]
                    clustSizeTmp[thisClust] = size
                    fitWithTmp[thisClust] = fitWithTmp[c1] + fitWithTmp[c2]
                    if clustersTmp[c2]:
                        # c2 is an actual cluster, not a virtual merger of 2 self-sufficient sub-clusters
                        fitWithTmp[thisClust].append(c2)
                    clustersTmp[c1] = None
                    clustSizeTmp[c1] = 0
                    fitWithTmp[c1] = None
                else:
                    # both c1 and c2 are self-sufficient, need to create a virtual merger of c1+c2 so that
                    # if later thisClust is used as fitWith for another cluster, we can do the right thing...
                    # A "virtual merger" has correct clustSize (so we'll know it's not a candidate for merging),
                    # and fitWithTmp contains the list of its non-virtual sub-clusters / components,
                    # but clustersTmp == None (the mark of a virtual merger)
                    clustSizeTmp[thisClust] = size
                    fitWithTmp[thisClust] = fitWithTmp[c1] + fitWithTmp[c2]
                    if clustersTmp[c1]:
                        fitWithTmp[thisClust].append(c1)
                    if clustersTmp[c2]:
                        fitWithTmp[thisClust].append(c2)
        else:
            # dist > maxDist, nothing more to do, just truncate the Tmp lists
            del clustersTmp[thisClust:]
            del clustSizeTmp[thisClust:]
            del fitWithTmp[thisClust:]
            break

    ################
    # step 3:
    # populate the clustering data structures from the Tmp lists, with proper formatting
    clust2samps = {}
    fitWith = {}
    clustIsValid = {}

    # populate clustIndex2ID with correctly constructed clusterIDs for each non-virtual cluster index
    clustIndex2ID = [None] * len(clustersTmp)
    nextClustNb = 1
    for thisClust in range(len(clustersTmp)):
        if clustersTmp[thisClust]:
            # left-pad with leading zeroes if less than 2 digits (for pretty-sorting, won't
            # sort correctly if more than 100 clusters but it's just cosmetic)
            clustID = chromType + '_' + f"{nextClustNb:02}"
            clustIndex2ID[thisClust] = clustID
            nextClustNb += 1
            clust2samps[clustID] = [samples[i] for i in clustersTmp[thisClust]]
            if fitWithTmp[thisClust]:
                fitWith[clustID] = [clustIndex2ID[i] for i in fitWithTmp[thisClust]]
            else:
                fitWith[clustID] = []
            if (clustSizeTmp[thisClust] >= minSamps):
                clustIsValid[clustID] = True
            else:
                clustIsValid[clustID] = False

    return (clust2samps, fitWith, clustIsValid)
