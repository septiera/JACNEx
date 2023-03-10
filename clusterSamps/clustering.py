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
# Returns a tuple (clust2Samps, trgt2Ctrls):
# - clust2Samps (dict(int : list[int])): clusterID associated to SOIsIndex
#   key = clusterID, value = list of SOIsIndex
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
def clustersBuilds(FPMarray, maxCorr, minCorr, minSamps, outputFile):
    # - minDist (float): is the distance to start cluster construction
    minDist = (1 - maxCorr)**0.5
    # - maxDist (float): is the distance to finalise the cluster construction
    maxDist = (1 - minCorr)**0.5

    # - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
    #  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
    linksMatrix = computeSampLinks(FPMarray)

    (clust2Samps, trgt2Ctrls) = links2Clusters(linksMatrix, minDist, maxDist, minSamps)

    # Optionnal plot a dendogram based on clustering results
    figures.plots.plotDendogram(clust2Samps, trgt2Ctrls, linksMatrix, minDist, outputFile)

    return(clust2Samps, trgt2Ctrls)


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
# parse the linkage matrix produced by computeSampLinksPrivate,
# clusters formation and identification of controls/targets clusters
# Conditions for building a cluster:
# 1- The distance between samples in the cluster must be within a range,
# specified by the minDist and maxDist parameters.
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
#  matrix, dim = (NbVSOIs-1)*[clusterID1,clusterID2,distValue,NbVSOIsInClust]
# - minDist (float): is the distance to start cluster construction
# - maxDist (float): is the distance to stop cluster construction
# - minSamps (int): minimal sample number to validate a cluster
#
# Returns a tupple (clust2Samps, trgt2Ctrls), each is created here:
# - clust2Samps (dict(int : list[int])): clusterID associated to valid sample indexes
#   key = clusterID, value = list of valid sample indexes
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
def links2Clusters(linksMatrix, minDist, maxDist, minSamps):
    # To Fill and returns
    clust2Samps = {}
    trgt2Ctrls = {}

    # To Fill, not returns
    # - links2Clusters (dict(int : list[int])): cluster formed from linksMatrix
    #   parsing associated with VSOIs.
    #   key = current clusterID, value = list of valid SOIs indexes
    # It is important not to change it to keep the patient lists healthy.
    # Hence the creation of a second dictionary respecting the conditions for
    # the construction of clusters "clust2Samps".
    links2Clusters = {}

    # To increment
    # - clusterID [int]: cluster identifiers
    # e.g first formed samples cluster => NbRow from linksMatrix +1 (e.q NbVSOIs)
    clusterID = len(linksMatrix)
    
    for clusterLine in linksMatrix:
        ##########################
        # PARSE LINKS MATRIX
        ##########
        clusterID += 1

        # keep parent clusters ID and convert it to int for extract easily VS indexes
        parentClustsIDs = [np.int(clusterLine[0]), np.int(clusterLine[1])]

        distValue = clusterLine[2]

        # parentsSOIsIndexes (list[int]): parent clusters VSOIs indexes
        # parentsSOIsNB (list[int]): samples number in each parent clusters
        (parentsSOIsIndexes, parentsSOIsNB) = getParentsClustsInfos(parentClustsIDs, links2Clusters, len(linksMatrix))

        ################
        # Fill links2Clusters
        links2Clusters[clusterID] = parentsSOIsIndexes

        """
        ################
        # DEV CONTROL ; check that the sample number for a cluster is correct
        # TO REMOVE
        NbSOIsInClust = clusterLine[3]
        if (len(VSOIsIndexInParents) != NbSOIsInClust):
            break
        """

        ##########################
        # CONDITIONS FOR CLUSTER CREATION
        # Populate "clusters" and "trgt2Ctrls"
        ##########
        # condition 1: cluster constructions from a minimum to a maximum correlation distance
        # replacement/overwriting of old clusterIDs with the new one for SOIs indexes in
        # np.array "clusters"
        # allows to build the clusters step by step in the order of correlations.
        # allows to keep the small clusters after the last threshold maxDists.
        # all samples are taken
        if (distValue < minDist):
            clust2Samps[clusterID] = parentsSOIsIndexes
            # deletion of old groups when forming new ones
            for key in parentClustsIDs:
                if key in clust2Samps:
                    del clust2Samps[key]

        # current distance is between minDist and maxDist
        # cluster selection is possible
        # From this step onwards, clusters with sufficient samples will be kept and will
        # be defined as controls for other clusters (fill trgt2Ctrls dictionnary)
        # the deletion of old clusters should therefore be treated with caution.
        elif ((distValue >= minDist) and (distValue <= maxDist)):
            ##########
            # condition 2: estimate the samples number to create a cluster
            # sufficient samples in current cluster
            if (len(parentsSOIsIndexes) >= minSamps):

                ###############
                # Different cases to complete the two dictionaries
                # Knowing that we are dealing with two parent clusters
                # Case 1: both parent clusters have sufficient numbers to be controls
                if ((parentsSOIsNB[0] >= minSamps) and (parentsSOIsNB[1] >= minSamps)):

                    # parent clusters are previously saved in clust2samps
                    if parentClustsIDs[0] in clust2Samps and parentClustsIDs[1] in clust2Samps:
                        # fill trgt2Ctrl
                        trgt2Ctrls[clusterID] = parentClustsIDs
                        # case that parentClustIDs are already controlled
                        for parentID in parentClustsIDs:
                            if parentID in trgt2Ctrls:
                                trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID] + trgt2Ctrls[parentID]
                    
                    # parent clusterID not in clust2samps
                    else:
                        clust2Samps[clusterID] = parentsSOIsIndexes

                # Case 2: one parent has a sufficient number of samples not the second parent
                elif max(parentsSOIsIndexes) >= 20:
                    # identification of the control parent and the target parent
                    # index corresponding to nbVSOIsInParents and parentClustsIDs (e.g list: [parent1, parent2])
                    indexCtrlParent = np.argmax(parentsSOIsNB)  # control
                    indexNewParent = np.argmin(parentsSOIsNB)  # target

                    if parentClustsIDs[indexCtrlParent] in clust2Samps.keys():
                        # fill trgt2Ctrl
                        # case that the parent control has previous controls
                        if (parentClustsIDs[indexCtrlParent] in trgt2Ctrls.keys()):
                            trgt2Ctrls[clusterID] = trgt2Ctrls[parentClustsIDs[indexCtrlParent]] + [parentClustsIDs[indexCtrlParent]]
                        else:
                            trgt2Ctrls[clusterID] = [parentClustsIDs[indexCtrlParent]]

                        # fill clust2Samps
                        # Keep only VSOIs index from the parent with few sample
                        if (indexNewParent == 0):
                            clust2Samps[clusterID] = parentsSOIsIndexes[:parentsSOIsNB[indexNewParent]]
                        else:
                            clust2Samps[clusterID] = parentsSOIsIndexes[-parentsSOIsNB[indexNewParent]:]
                    
                    # case where the parent cluster comes from two clusters with sufficient samples
                    # not present in clusts2Samps but in trgt2Ctrls
                    elif parentClustsIDs[indexCtrlParent] in trgt2Ctrls.keys():
                        clust2Samps[clusterID] = parentsSOIsIndexes[:parentsSOIsNB[indexNewParent]]
                        trgt2Ctrls[clusterID] = trgt2Ctrls[parentClustsIDs[indexCtrlParent]]
                    else:
                        clust2Samps[clusterID] = parentsSOIsIndexes

                    if parentClustsIDs[indexNewParent] in clust2Samps:
                        del clust2Samps[parentClustsIDs[indexNewParent]]

                # Case 3: each parent cluster has an insufficient number of samples.
                # the current clusterID becomes a control cluster.
                else:
                    clust2Samps[clusterID] = parentsSOIsIndexes
                    if parentClustsIDs[0] in clust2Samps:
                        del clust2Samps[parentClustsIDs[0]]
                    if parentClustsIDs[1] in clust2Samps:
                        del clust2Samps[parentClustsIDs[1]]

            # clusters too small
            # complete clust2Samps with new clusterID and remove old clustersID
            else:
                clust2Samps[clusterID] = parentsSOIsIndexes
                for key in parentClustsIDs:
                    if key in clust2Samps:
                        del clust2Samps[key]

        # current distance larger than maxDist we stop the loop on linksMatrix rows
        elif (distValue > maxDist):
            break
    return(clust2Samps, trgt2Ctrls)


#############################
# getParentsClustsInfos
# function used in another private function links2ClustersFormationPrivate
# Extract parents informations : SOIs indexes list and samples number
#
# Args:
# - parentClustsIDs (list[int]): two parents clusters identifiers to be combined
# - links2Clusters (dict(int : list[int])): cluster formed thanks to the linksMatrix
#   parsing associated with VSOIs.
#   key = current clusterID, value = list of VSOIs indexes
# - nbLinks (int): links number in linksMatrix (row count)
#
# Returns a tuple (parentsSOIsIndexes, parentsSOIsNB), each is created here:
# - parentsSOIsIndexes (list[int]): parent clusters VSOIs indexes
# - parentsSOIsNB (list[int]): samples number in each parent clusters
def getParentsClustsInfos(parentClustsIDs, links2Clusters, NbLinks):
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
            parentsSOIsIndexes = parentsSOIsIndexes + links2Clusters[parentID]
            parentsSOIsNB.append(len(links2Clusters[parentID]))
    return(parentsSOIsIndexes, parentsSOIsNB)
