import logging
import numpy as np

# different scipy submodules are used for the application of hierachical clustering
import scipy.cluster.hierarchy
import scipy.spatial.distance

import figures.plots

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################
# clustersBuilds
# Groups the QC validated samples according to their distance obtained
# with a hierarchical clustering.
# Transforms the correlation thresholds (maxCorr, minCorr = ρ) passed as arguments into
# distance (√(1-ρ)).
# Compute distances between samples with the equation √(1-ρ) where ρ is
# the Pearson correlation.
# Calculate matrix links with hierarchical clustering using the 'average'
# method on the distance data.
# Parsing this links matrix allows to obtain the clusters but respecting some conditions:
# 1- The distance between samples in the cluster must be within a range,
# specified by the minDist and maxDist parameters.
# 2- The number of samples in the cluster must be above a threshold,
# specified by the minSamps parameter.
# Identification of control clusters (samples composing a cluster with small distances
# between them and with sufficient numbers) used as reference for the formation of
# other clusters (target clusters, present for higher distances)
# returns a pdf in the plotDir.
#
# Args:
#  - FPMarray (np.ndarray[float]): normalised fragment counts for QC validated samples,
#  dim = NbCapturedExons x NbSOIsQCValidated
#  - maxCorr (float): maximal Pearson correlation score tolerated by the user to start
#   build clusters
#  - minCorr (float): minimal Pearson correlation score tolerated by the user to end
#   build clusters
#  - minSamps (int): minimal sample number to validate a cluster
#  - outputFile (str): full path (+ file name) for saving a dendogram
#
# return a string list of list of dim = nbClust * ["clustID", "SampsInCluster", "validCluster"]
def clustersBuilds(FPMarray, SOIs, maxCorr, minCorr, minSamps, outputFile):
    # - minDist (float): is the distance to start cluster construction
    minDist = (1 - maxCorr)**0.5
    # - maxDist (float): is the distance to finalise the cluster construction
    maxDist = (1 - minCorr)**0.5

    # - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
    #  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
    linksMatrix = computeSampLinks(FPMarray)

    clusters = links2Clusters(SOIs, linksMatrix, minDist, maxDist, minSamps)

    # Optionnal plot a dendogram based on clustering results
    figures.plots.plotDendogram(clusters, linksMatrix, minDist, outputFile)

    return(clusters)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################
# computeSampLinks
# Pearson correlation distance (sqrt(1-r)) is unlikely to be a sensible
# distance when clustering samples.
# (sqrt(1-r)) is a true distance respecting symmetry, separation and triangular
# inequality
# average linkage method is the best choice when there are different-sized groups
#
# Args:
# - FPMarray (np.ndarray[float]): normalised fragment counts for QC validated
#  samples (VSOIs), dim = NbCapturedExons x NbSOIsQCValidated
#
# Return:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbVSOIs-1)*[clusterID1,clusterID2,distValue,NbVSOIsInClust]
def computeSampLinks(FPMarray):

    correlation = np.round(np.corrcoef(FPMarray, rowvar=False), 2)
    dissimilarity = (1 - correlation)**0.5

    # "squareform" transform squared distance matrix in a triangular matrix
    # "optimal_ordering": linkage matrix will be reordered so that the distance between
    # successive leaves is minimal.
    linksMatrix = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'average', optimal_ordering=True)
    return(linksMatrix)


#############################
# links2Clusters
# parse the linkage matrix produced by computeSampLinks,
# clusters formation and identification of controls/targets clusters
# Conditions for building a cluster:
# 1- The distance between samples in the cluster must be less than maxDist.
# 2- The number of samples in the cluster must be above a threshold,
# specified by the minSamps parameter.
# a cluster called control has for attribute:
#  -contains enough samples (>=20)
#  -formed at very small distances (generally as soon as minDist)
#  -the samples of this cluster are used to form another cluster with more
# distant samples. The cluster then formed is called target.
#
# Args:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
# - minDist (float): is the distance to start cluster construction
# - maxDist (float): is the distance to stop cluster construction
# - minSamps (int): minimal sample number to validate a cluster
#
# return a string list of list of dim = nbClust * ["clustID", "SampsInCluster", "validCluster"]

def links2Clusters(linksMatrix, SOIs, minDist, maxDist, minSamps):
    # To Fill and not returns
    # - clust2Samps (dict(int : list[int])): clusterID associated to sample indexes
    #   key = clusterID, value = list of sample indexes
    # only add indexed SOIs are contained in the list (no SOIs indexes from the control cluster)
    clust2Samps = {}
    # - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
    #   key = target clusterID, value = list of controls clusterID
    trgt2Ctrls = {}

    # To increment
    # - clusterID [int]: cluster identifiers
    # e.g first formed samples cluster => NbRow from linksMatrix +1 (e.q NbSOIs)
    clusterID = len(linksMatrix)

    for currentCluster in linksMatrix:

        clusterID += 1
        ##########################
        # PARSE LINKS MATRIX
        ##########
        # keep parent clusters ID and convert it to int for extract easily SOIs indexes
        parentClustsIDs = [np.int(currentCluster[0]), np.int(currentCluster[1])]

        distValue = currentCluster[2]

        # clustSOIsIndexes (list[int]): cluster SOIs indexes (from parents clusters)
        # parentsSOIsNB (list[int]): samples number in each parent clusters
        (clustSOIsIndexes, parentsSOIsNB) = getParentsClustsInfos(parentClustsIDs, clust2Samps, len(linksMatrix))

        ################
        # DEV CONTROL ; check that the sample number for a cluster is correct
        # TO REMOVE
        NbSOIsInClust = currentCluster[3]
        if (len(clustSOIsIndexes) != NbSOIsInClust):
            break

        ##########################
        # CONDITIONS FOR CLUSTER CREATION
        # Populate "clust2Samps" and "trgt2Ctrls"
        ##########
        # condition for creating valid clusters
        # - the distance is between minDist and maxDist
        # - the number of samples in current cluster is less than minSamps
        # if not, create a new cluster and remove the parents in clust2Samps.
        # This allows the most correlated clusters to be constructed first,
        # even if they don't meet the conditions.
        if (distValue < minDist) or (len(clustSOIsIndexes) < minSamps):
            clust2Samps[clusterID] = clustSOIsIndexes
            for key in parentClustsIDs:
                if key in clust2Samps:
                    del clust2Samps[key]
            continue
        # current distance larger than maxDist we stop the loop on linksMatrix rows
        # avoids going through the rest of the clusters
        elif (distValue > maxDist):
            break

        ###############
        # Different cases to complete the two dictionaries
        # Knowing that we are dealing with two parent clusters
        # Case 1: both parent clusters have sufficient numbers to be controls
        if ((parentsSOIsNB[0] >= minSamps) and (parentsSOIsNB[1] >= minSamps)):            
            # parent clusters are in clust2samps
            if (parentClustsIDs[0] in clust2Samps) and (parentClustsIDs[1] in clust2Samps):
                # fill trgt2Ctrl
                trgt2Ctrls[clusterID] = parentClustsIDs
                # case that parentClustIDs are already controlled
                for parentID in parentClustsIDs:
                    if parentID in trgt2Ctrls:
                        trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID].copy() + trgt2Ctrls[parentID].copy()

            # parent clusterID not in clust2samps
            else:
                # fill clust2Samps
                clust2Samps[clusterID] = clustSOIsIndexes

        # Case 2: one parent has a sufficient number of samples not the second parent
        elif max(parentsSOIsNB) >= 20:
            # identification of the control parent and the target parent
            # index corresponding to nbSOIsInParents and parentClustsIDs (e.g list: [parent1, parent2])
            indexCtrlParent = np.argmax(parentsSOIsNB)  # control
            indexNewParent = np.argmin(parentsSOIsNB)  # target

            # parent cluster control are in clust2samps
            if parentClustsIDs[indexCtrlParent] in clust2Samps.keys():
                # fill trgt2Ctrl
                # case that the parent control has previous controls
                if (parentClustsIDs[indexCtrlParent] in trgt2Ctrls.keys()):
                    trgt2Ctrls[clusterID] = trgt2Ctrls[parentClustsIDs[indexCtrlParent]].copy() + [parentClustsIDs[indexCtrlParent]].copy()
                else:
                    trgt2Ctrls[clusterID] = [parentClustsIDs[indexCtrlParent]]

                # fill clust2Samps
                # Keep only SOIs index from the parent with few sample (target)
                if (indexNewParent == 0):
                    clust2Samps[clusterID] = clustSOIsIndexes[:parentsSOIsNB[indexNewParent]]
                else:
                    clust2Samps[clusterID] = clustSOIsIndexes[-parentsSOIsNB[indexNewParent]:]

            # the parent cluster control comes from two control clusters
            # not present in clusts2Samps but in trgt2Ctrls (update them)
            elif parentClustsIDs[indexCtrlParent] in trgt2Ctrls.keys():
                clust2Samps[clusterID] = clustSOIsIndexes[:parentsSOIsNB[indexNewParent]]
                trgt2Ctrls[clusterID] = trgt2Ctrls[parentClustsIDs[indexCtrlParent]].copy()

            # the parent cluster control not exist in clusts2Samps and in trgt2Ctrls
            # is formed with small distances so not referenced
            else:
                clust2Samps[clusterID] = clustSOIsIndexes

            # deleting the cluster with few samples
            if parentClustsIDs[indexNewParent] in clust2Samps:
                del clust2Samps[parentClustsIDs[indexNewParent]]

        # Case 3: each parent cluster has an insufficient number of samples.
        # the current clusterID becomes a valid cluster.
        else:
            clust2Samps[clusterID] = clustSOIsIndexes
            if parentClustsIDs[0] in clust2Samps:
                del clust2Samps[parentClustsIDs[0]]
            if parentClustsIDs[1] in clust2Samps:
                del clust2Samps[parentClustsIDs[1]]

    # the transformation of results is carried out here to avoid using (non-stable)
    # dictionaries outside their definition function
    try:
        clusters = parseResults(SOIs, clust2Samps, trgt2Ctrls, minSamps)
    except Exception as e:
        raise

    return(clusters)


#############################
# getParentsClustsInfos
# function used by links2Clusters function
# Extract for each parents : SOIs indexes list and samples number
#
# Args:
# - parentClustsIDs (list[int]): two parents clusters identifiers to be combined
# - clust2Samps (dict(int : list[int])): cluster formed thanks to the linksMatrix
#   parsing associated with VSOIs.
#   key = current clusterID, value = list of SOIs indexes
# - nbLinks (int): links number in linksMatrix (row count)
#
# Returns a tuple (parentsSOIsIndexes, parentsSOIsNB), each is created here:
# - parentsSOIsIndexes (list[int]): parent clusters SOIs indexes
# - parentsSOIsNB (list[int]): samples number in each parent clusters
def getParentsClustsInfos(parentClustsIDs, clust2Samps, NbLinks):
    parentsSOIsIndexes = []
    parentsSOIsNB = []

    for parentID in parentClustsIDs:
        #####
        # where it's a sample identifier not a cluster
        # the clusterID corresponds to the SOI index
        if (parentID <= NbLinks):
            parentsSOIsIndexes.append(parentID)
            parentsSOIsNB.append(1)
        #####
        # where it's a cluster identifier
        # we get indexes lists
        else:
            parentsSOIsIndexes = parentsSOIsIndexes + clust2Samps[parentID].copy()
            parentsSOIsNB.append(len(clust2Samps[parentID]))
    return(parentsSOIsIndexes, parentsSOIsNB)


###########################
# parseResults
# formatting the clustering results
# generalisation of cluster identifiers (from 1 to nbclusters)
# invalid cluster identification => number of samples less than minDist
# identifications of samples that could not be clustered placed in the
# output end as "Samps_ClustFailed"
#
# Args:
# - SOIs (list[str]): samples of interest
# - clust2Samps (dict([int]:list[int])): a dictionary mapping cluster IDs to lists of samples
# indexes in those clusters
# - trgt2Ctrls (dict([int]:list[int])): a dictionary mapping target cluster IDs to lists of
# control cluster IDs.
# - minSamps [int]: the minimum number of samples required for a cluster to be considered valid
# return a string list of list of dim = nbClust * ["clustID", "SampsInCluster", "validCluster"]
def parseResults(SOIs, clust2Samps, trgt2Ctrls, minSamps):
    # To Fill and return
    resList = []

    # To Fill and not return
    # string list of sample names included in a cluster (valid or not)
    totalSOIs = []
    listToFill = []

    # extraction of the key list of clust2Samps to use the indices of each clusterID
    # as new clusterID
    clustersIDList = list(clust2Samps.keys())

    for cluster in clustersIDList:
        # Fixed parameters
        ctrlList = []  # list of new control clustID
        clusterValidity = 1

        # correspondence between sample names and sample indexes
        listSOIs = [SOIs[i] for i in clust2Samps[cluster]]

        # case of controlled target cluster replacement of clustID of controls
        if cluster in trgt2Ctrls:
            for ctrl in trgt2Ctrls[cluster]:
                ctrlList.extend(str(clustersIDList.index(ctrl)))
        # case where the cluster is not controlled
        # validated test => nb sample must be greater than minSamps
        else:
            if len(listSOIs) < minSamps:
                logger.info("cluster n°%s : does not contain enough sample to be valid (%s)", clustersIDList.index(cluster), len(listSOIs))
                clusterValidity = 0

        # Fill resList
        listToFill = [str(clustersIDList.index(cluster)), ",".join(listSOIs), ",".join(ctrlList), clusterValidity]
        resList.append(listToFill)

        totalSOIs.extend(listSOIs)

    # test 1: each sample must be included in a single cluster
    if len(totalSOIs) != len(set(totalSOIs)):
        duplicateSamps = list(set([item for item in totalSOIs if totalSOIs.count(item) > 1]))
        logger.info("Sample(s) %s is(are) contained in two clusters which is not allowed.", ",".join(duplicateSamps))
        raise Exception('sample contained in several reference clusters')
    # test 2: some samples do not cluster because they are too distant
    if len(totalSOIs) != len(SOIs):
        failedSampsClust = set(SOIs).difference(set(totalSOIs))
        logger.info("Samples %s are too far away from the other samples to be associated with a cluster.", ",".join(failedSampsClust))
        listToFill = ["Samps_ClustFailed", ",".join(failedSampsClust), "", ""]
        resList.append(listToFill)

    return(resList)
