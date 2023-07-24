import logging
import numpy as np
import os

# different scipy submodules are used for the application of hierachical clustering
import scipy.cluster.hierarchy
import scipy.spatial.distance

import figures.plots

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################
# clustersBuilds
# Groups the samples according to their distance obtained with a hierarchical clustering.
# Transforms the correlation thresholds (maxCorr, minCorr = ρ) passed as arguments into
# distance (√(1-ρ)).
# Compute distances between samples with the equation √(1-ρ) where ρ is
# the Pearson correlation.
# Calculate matrix links with hierarchical clustering using the 'average'
# method on the distance data.
# Parsing this links matrix and obtaining the clusters.
# Formats the results by checking their validity
# Produces a dendrogram in plotDir
#
# Args:
#  - FPMarray (np.ndarray[float]): normalised fragment counts for samples,
#  dim = NbCapturedExons x NbSOIs
#  - maxCorr (float): maximal Pearson correlation score tolerated by the user to start
#   build clusters
#  - minCorr (float): minimal Pearson correlation score tolerated by the user to end
#   build clusters
#  - minSamps (int): minimal sample number to validate a cluster
#  - plotFile (str): full path (+ file name) for saving a dendrogram
#
# Returns (clusters, linksMatrix):
#  - clusters (list of list[int, list[int], list[int], int]): cluster definitions
# dim = nbClust * ["CLUSTER_ID","SAMPLES","CONTROLLED_BY","VALIDITY"]
#  - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSamples-1)*[clusterID1,clusterID2,distValue,NbSamplesInClust]
# this output is only used for testing and debugging
def clustersBuilds(FPMarray, maxCorr, minCorr, minSamps, plotFile):
    if os.path.isfile(plotFile):
        logger.error('clustering dendrogram : plotFile %s already exist', plotFile)
        raise Exception("plotFile already exist")

    # clustering method
    CM = "average"

    # To Fill and returns
    clusters = []

    # - minDist (float): is the distance to start cluster construction
    minDist = (1 - maxCorr)**0.5
    # - maxDist (float): is the distance to finalise the cluster construction
    maxDist = (1 - minCorr)**0.5

    # hierarchical clustering
    linksMatrix = computeSampsLinks(FPMarray, CM)

    samps2Clusters, trgt2Ctrls = links2Clusters(linksMatrix, minDist, maxDist, minSamps)
    clustsList = np.unique(samps2Clusters)

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

    figures.plots.plotDendrogram(linksMatrix, labelsGp, minDist, CM, plotFile)

    return (clusters, linksMatrix)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################
# computeSampsLinks
# Pearson correlation distance (sqrt(1-r)) is likely to be a sensible
# distance when clustering samples.
# (sqrt(1-r)) is a true distance respecting symmetry, separation and triangular
# inequality
# average linkage method is the best choice when there are different-sized groups
#
# Args:
# - FPMarray (np.ndarray[float]): normalised fragment counts, dim = NbCapturedExons x NbSamples
# - CM [str]: clustering method
# Returns:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSamples-1)*[clusterID1,clusterID2,distValue,NbSamplesInClust]
def computeSampsLinks(FPMarray, CM):

    correlation = np.round(np.corrcoef(FPMarray, rowvar=False), 2)
    dissimilarity = (1 - correlation)**0.5

    # "squareform" transform squared distance matrix in a triangular matrix
    # "optimal_ordering": linkage matrix will be reordered so that the distance between
    # successive leaves is minimal.
    linksMatrix = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), CM, optimal_ordering=True)
    return(linksMatrix)


#############################
# links2Clusters
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
