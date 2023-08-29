import logging
import numpy as np
import os

# modules for hierachical clustering
import scipy.cluster.hierarchy
import scipy.spatial.distance
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
# - chromType (string): one of 'A', 'XZ', 'YW' indicating that FPMarray holds counts
#   for exons on autosomes, X (or Z) chromosome, or Y (or W) chromosome, respectively
# - samples: list of sampleIDs, same order as the columns of FPMarray
# - startDist: smallest distance at which we start trying to build clusters
# - maxDist: max distance up to which we can build clusters / populate FIT_WITH
# - minSamps: min number of samples (in a cluster + its FIT_WITH friends) to declare
#   the cluster VALID
# - plotFile (str): filename (including path) for saving the dendrogram representing
#   the resulting hierarchical clustering, use "" if you don't want a plot
#
# Returns (clust2samps, fitWith, clustIsValid, linkageMatrix):
# (clust2samps, fitWith, clustIsValid) are as defined in clustFile.py parseClustsFile(), ie
# clusterIDs are formatted as TYPE_NUMBER, where TYPE is 'A', 'XZ' or 'YW', and:
# - clust2samps: dict, key==clusterID, value == list of sampleIDs
# - fitWith: dict, key==clusterID, value == list of clusterIDs
# - clustIsValid: dict, key==clusterID, value == Boolean
# - linkageMatrix (provided only for testing and debugging) is the linkage matrix
#   encoding the hierarchical clustering as returned by scipy.cluster.hierarchy.linkage,
#   ie each row corresponds to one merging step of 2 cluster (c1, c2) and holds
#   [c1, c2, dist(c1,c2), size(c1) + size(c2)]
def buildClusters(FPMarray, chromType, samples, startDist, maxDist, minSamps, plotFile):
    if (plotFile != "") and os.path.isfile(plotFile):
        logger.warning("buildClusters can't produce a dendrogram: plotFile %s already exists",
                       plotFile)
        plotFile = ""

    # reduce dimensionality with PCA
    # we don't really want the smallest possible number of dimensions, try arbitrary
    # smallish dims (must be < nbExons and < nbSamples)
    dim = min(10, FPMarray.shape[0], FPMarray.shape[1])
    samplesInPCAspace = sklearn.decomposition.PCA(n_components=dim).fit_transform(FPMarray.transpose())

    # hierarchical clustering of the samples projected in the PCA space:
    # - use 'average' method to define the distance between clusters (== UPGMA),
    #   not sure why but AS did a lot of testing and says it is the best choice
    #   when there are different-sized groups;
    # - use 'euclidean' metric to define initial distances between samples;
    # - reorder the linkage matrix so the distance between successive leaves is
    #   minimal [NOT sure about this one, documentation says it slows down things a lot
    #   and we can do some sorting within dendrogram()].
    linkageMatrix = scipy.cluster.hierarchy.linkage(samplesInPCAspace, method='average',
                                                    metric='euclidean', optimal_ordering=True)

    # build clusters from the linkage matrix
    (clust2samps, fitWith, clustIsValid) = linkage2clusters(linkageMatrix, chromType, samples,
                                                            startDist, maxDist, minSamps)

    # produce and plot dendrogram
    if (plotFile != ""):
        makeDendrogram(linkageMatrix, clust2samps, fitWith, clustIsValid, startDist, plotFile)

    # sort samples in clust2samps and clusters in fitWith (for cosmetics)
    for clust in clust2samps:
        clust2samps[clust].sort()
        fitWith[clust].sort()
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
# - startDist, maxDist, minSamps: same args as buildClusters(), used for constructing
#   the clusters
#
# Returns (clust2samps, fitWith, clustIsValid) as specified in the header of buildClusters()
def linkage2clusters(linkageMatrix, chromType, samples, startDist, maxDist, minSamps):
    numSamples = len(samples)
    ################
    # step 1:
    # - populate clustersTmp, a list of lists of ints:
    #   clustersTmp[i] is the list of sample indexes that belong to cluster i (at this step),
    #   this will be cleared at a later step if i is merged with another cluster
    # NOTE I need to preallocate so I can use clustersTmp[i] = "toto"
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
    # step 2:
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
            # sort correctly if more than 100 clusters but it's just cosmetic
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


#############################
# TODO: write spec for makeDendrogram, which apparently needs to build some kind
# of "labels" and prepare some stuff, then call figures.plots.plotDendrogram()
# Below I just copy-pasted AS's code that she had in the main buildClusters() function.
# I'm a bit confused because her code seems to do more than just dendrogram prep.
# Actually I expect makeDendrogram should actually be pretty simple, just some small cosmetic
# prep for plotDendrogram()...
# I'll have to really dig into it, no time now
def makeDendrogram(linkageMatrix, clust2samps, fitWith, clustIsValid, startDist, plotFile):
    # To Fill not returns
    # labelArray (np.ndarray[str]): labels for each sample within each cluster, dim=NbSOIs*NbClusters
    labelArray = np.empty([len(samps2Clusters), len(clustsList)], dtype="U1")
    labelArray.fill("")
    # labelsGp (list[str]): labels for each sample list to be passed when plotting the dendrogram
    labelsGp = []

    # labels definitions for dendrogram
    label1 = "x"  # sample contributes to the cluster
    label2 = "-"  # sample controls the cluster
    label3 = "O"  # sample is not successfully clustered

    #######
    # results formatting and preparation of the graphic representation
    # replacement of cluster identifiers by decreasing correlation order
    # validity check based on counts and if all samples were clustered (0 invalid, 1 valid)
    for i in range(len(clustsList)):
        validityStatus = 1

        # keep samples indexes for current cluster
        sampsClustIndex = list(np.where(samps2Clusters == clustsList[i])[0])

        # associate the label to the sample index(column) and the cluster index position(row)
        labelArray[sampsClustIndex, i] = label1

        # list to store new control clusterID
        ctrlList = []
        CTRLsampsIndex = []
        # controlled cluster replacement of controls clusterID
        if clustsList[i] in trgt2Ctrls:
            for ctrl in trgt2Ctrls[clustsList[i]]:
                ctrlList.extend(list(np.where(clustsList == ctrl))[0])
                CTRLsampsIndex.extend(list(np.where(samps2Clusters == ctrl)[0]))

        labelArray[CTRLsampsIndex, i] = label2

        # check validity
        if ((len(sampsClustIndex) + len(CTRLsampsIndex)) < minSamps) or (clustsList[i] == 0):
            validityStatus = 0
            labelArray[sampsClustIndex, i] = label3

        listToFill = [i, sampsClustIndex, ctrlList, validityStatus]
        clusters.append(listToFill)

    # browse the np array of labels to build the str list (need for plot dendrogram)
    for i in labelArray:
        # separation of labels for readability
        strToBind = "  ".join(i)
        labelsGp.append(strToBind)

    title = "hierarchical clustering method='average' metric='euclidean' chromType='" + chromType + "'"
    figures.plots.plotDendrogram(linkageMatrix, labelsGp, startDist, title, plotFile)


#############################
# links2Clusters
# OLD CODE FROM AS, should be somewhat equivalent to linkage2clusters() although
# the code is shorter and it seems to do less (eg doesn't set clustIsValid)
# ... I think this code only does a first step of cluster construction, despite
# the function's name, and some of the functionality if linkage2clusters() is
# actually in the makeDendrogram() code above...
# WILL NEED TO DIG
#
# parse the linkage matrix produced by computeSampLinks,
# goals: construct the clusters in descending order of correlations
# Conditions for building a cluster:
# 1- The distance between samples in the cluster must be less than maxDist.
# 2- The number of samples in the cluster must be above a threshold,
# specified by the minSamps parameter.
# addition of a feature allowing the creation of target clusters that come
# from clusters with enough samples for short distances (called cluster controls)
#
# Args:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSamples-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
# - minDist (float): is the distance allowing to start the clusters controls creation
# - maxDist (float): is the distance to stop cluster construction
# - minSamps (int): minimal sample number to validate a cluster
#
# Returns a tupple (samps2Clusters, trgt2Ctrls), each is created here:
# - samps2Clusters (np.ndarray[int]): clusterID for each sample indexe
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
def links2Clusters(linksMatrix, minDist, maxDist, minSamps):
    # To fill and returns
    samps2Clusters = np.zeros(len(linksMatrix) + 1, dtype=int)
    trgt2Ctrls = {}

    # To increment
    # e.g first formed samples cluster => NbRow from linksMatrix +1 (e.q samplesNB)
    clusterID = len(linksMatrix)

    # parse linkage matrix rows => formation of clusters from each sample in increasing
    # order of distances (decreasing for correlations)
    for currentCluster in linksMatrix:
        clusterID += 1

        # storage list[int] for clusterID control if required
        # used as value to fill trgt2Ctrls dictionary
        controls = []

        parentClust1 = np.int(currentCluster[0])
        parentClust2 = np.int(currentCluster[1])
        distValue = currentCluster[2]

        # distance is too great, risk of integrating samples that are too far
        # or merging clusters with too different coverage profiles
        if (distValue > maxDist):
            break

        # applying different conditions for each parent to fill samps2Clusters
        # and trgt2Ctrls
        for parentID in [parentClust1, parentClust2]:

            # the parent cluster is a single sample
            # => fill the clusterID to the sample index
            # move to next parent or end of loop
            if parentID < len(linksMatrix) + 1:
                samps2Clusters[parentID] = clusterID
                continue

            # get indices from the samples of the parent cluster
            sampsParentIndex = list(np.where(samps2Clusters == parentID)[0])
            # the parent may already be controlled by another(s) cluster(s)
            if parentID in trgt2Ctrls:
                # the current cluster becomes a new control cluster
                # don't forget it when creating the new list of controls
                controls.extend(trgt2Ctrls[parentID])

                # expand sample list with samples from controls
                prevCtrl = trgt2Ctrls[parentID]
                for ctrl in prevCtrl:
                    sampsParentIndex.extend(list(np.where(samps2Clusters == ctrl)[0]))

            # we don't want to create a cluster definitively but replace the clusterIDs
            # for the samples indexes if the distance between the two parents is low
            # or if the number of samples in the current cluster is too low
            # move to next parent or end of loop
            if (distValue < minDist) or (len(sampsParentIndex) < minSamps):
                samps2Clusters[sampsParentIndex] = clusterID
                continue

            # the current parent cluster can be considered as a control,
            # we don't want to modify the indexes of the associated samples
            controls.extend([parentID])

        # filling of the dictionary if at least one of the two parents is
        # a control and/or has controls
        if len(controls) != 0:
            trgt2Ctrls[clusterID] = controls

    return (samps2Clusters, trgt2Ctrls)
