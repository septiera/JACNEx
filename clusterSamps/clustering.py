import logging
import numpy as np
import os

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
#  - plotFile (str): full path (+ file name) for saving a dendogram
#
# return a string list of list of dim = nbClust * ["clustID", "SampsInCluster", "validCluster"]
def clustersBuilds(FPMarray, maxCorr, minCorr, minSamps, plotFile):
    if os.path.isfile(plotFile):
        logger.error('clustering dendogramm : plotFile %s already exist', plotFile)
        raise Exception("plotFile already exist")

    # - minDist (float): is the distance to start cluster construction
    minDist = (1 - maxCorr)**0.5
    # - maxDist (float): is the distance to finalise the cluster construction
    maxDist = (1 - minCorr)**0.5

    # - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
    #  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
    linksMatrix = computeSampLinks(FPMarray)

    clusters = links2Clusters(linksMatrix, minDist, maxDist, minSamps)

    figures.plots.plotDendogram(clusters, linksMatrix, minDist, plotFile)

    return(clusters, linksMatrix)


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
#  samples, dim = NbCapturedExons x NbSamplesQCValidated
#
# Return:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSamples-1)*[clusterID1,clusterID2,distValue,NbSamplesInClust]
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
# 1- The number of samples in the cluster must be above a threshold,
# specified by the minSamps parameter.
# 2- The distance between samples in the cluster must be less than maxDist.
# a cluster called control has for attribute:
#  -contains enough samples (>=20 by default)
#  -formed at very small distances (generally as soon as minDist)
#  -the samples of this cluster are used to form another cluster with more
# distant samples. The cluster then formed is called target.
#
# Args:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSamples-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
# - minDist (float): is the distance to start cluster construction
# - maxDist (float): is the distance to stop cluster construction
# - minSamps (int): minimal sample number to validate a cluster
#
# return a list of list of dim = nbClust * ["newClustID","SampsInCluster","controlsClustersID"]
def links2Clusters(linksMatrix, minDist, maxDist, minSamps):
    # To Fill and not returns
    # as each line of linksMatrix corresponds to the association of samples in clusters
    # according to an increasing level of distance => cluster construction tracking,
    # facilitates the search for parent clusters.
    # must not be modified
    # keys: clusterID (from linksMatrix), value: cluster samples
    clust2Samps = {}

    # a list of int containing all cluster identifiers passing the filters:
    # nbsamples >= minSamps and distvalue<maxDist.
    validClustsIDs = []
    # list of sample lists for each cluster present in validClustsIDs
    validClustsSamps = []

    # key : clusterID of the cluster (target) formed from another cluster,
    # value: list of the reference cluster(s)
    trgt2Ctrls = {}

    # numpy boolean array to check that all samples are in valid clusters
    # (0: clustering failed, 1: clustering valid), dim = samplesNB
    checkClusteredSamps = np.zeros(len(linksMatrix) + 1, dtype=np.bool_)

    # To increment
    # - clusterID [int]: cluster identifiers
    # e.g first formed samples cluster => NbRow from linksMatrix +1 (e.q samplesNB)
    clusterID = len(linksMatrix)

    for currentCluster in linksMatrix:
        clusterID += 1
        ##########################
        # PARSE LINKS MATRIX
        ##########
        # keep parent clusters ID and convert it to int for extract easily samples indexes
        parentClustsIDs = [np.int(currentCluster[0]), np.int(currentCluster[1])]

        # If the key does not exist, insert the key, with the specified value
        # is only possible when adding a single sample
        for p in parentClustsIDs:
            clust2Samps.setdefault(p, [p])

        distValue = currentCluster[2]

        # extraction of the list of sample indexes from both parents
        # important for filling validClustSamps
        clustSampsIndexes = clust2Samps[parentClustsIDs[0]].copy()
        clustSampsIndexes.extend(clust2Samps[parentClustsIDs[1]].copy())

        # ################
        # # DEV CONTROL ; check that the sample number for a cluster is correct
        # # TO REMOVE
        # NbSampsInClust = currentCluster[3]
        # if (len(clustSampsIndexes) != NbSampsInClust):
        #     logger.error("Not Same samples in current cluster %s and parents clusters %s", clusterID, ",".join(parentClustsIDs))
        #     break

        # Populate "clust2Samps"
        clust2Samps[clusterID] = clustSampsIndexes

        ##########################
        # CONDITIONS FOR CLUSTER CREATION
        # populate "trgt2Ctrls", "validClustsIDs", "validClustsSamps", "clustSamps"
        ##########
        # condition for creating valid clusters
        # - the number of samples to validate a cluster must be greater than minSamps
        # stringent condition
        if (len(clustSampsIndexes) < minSamps):
            continue

        # - the distance must be between minDist and maxDist to create clusters.
        # However flexibility is tolerated if distValue is lower than minDist
        # => allows to recover the most correlated clusters.
        # Overwrite the latter in case they form other clusters under the minDist threshold.
        if (distValue < minDist):
            validClustsIDs.append(clusterID)
            validClustsSamps.append(clustSampsIndexes)
            checkClusteredSamps[clustSampsIndexes] = 1
            for p in parentClustsIDs:
                if p in validClustsIDs:
                    indexToreplace = validClustsIDs.index(p)
                    del validClustsIDs[indexToreplace]
                    del validClustsSamps[indexToreplace]
            continue
        # current distance larger than maxDist we stop the loop on linksMatrix rows
        # avoids going through the rest of the cluster lines
        elif (distValue > maxDist):
            break

        # here the clusters have a distance between minDist and maxDist and the number of
        # their samples is greater than minSamps
        # several conditions for a clear storage of the clustering information have to be respected:
        # 1) - neither parent clusters identifiers is in the validClustsIDs list
        # in some cases the current cluster has for parent a cluster formed by two valid clusters
        # so it's not in validClustsIDs but in trgt2Ctrls => update trgt2Ctrls
        # other case both parents are not in validClustIDs nor in trgt2Ctrls validation of current cluster
        if (parentClustsIDs[0] not in validClustsIDs) and (parentClustsIDs[1] not in validClustsIDs):
            if parentClustsIDs[0] not in trgt2Ctrls and parentClustsIDs[1]not in trgt2Ctrls:
                validClustsIDs.append(clusterID)
                validClustsSamps.append(clustSampsIndexes)
                checkClusteredSamps[clustSampsIndexes] = 1
            else:
                trgt2Ctrls[clusterID] = [parentClustsIDs[0], parentClustsIDs[1]]
                for p in parentClustsIDs:
                    if p in trgt2Ctrls:
                        trgt2Ctrls[clusterID] = trgt2Ctrls[p].copy() + trgt2Ctrls[clusterID].copy()
                    else:
                        continue

        # 2)- both parent clusters identifiers are in the validClustsIDs list
        # fill trgt2Ctrls
        elif (parentClustsIDs[0] in validClustsIDs) and (parentClustsIDs[1] in validClustsIDs):
            trgt2Ctrls[clusterID] = [parentClustsIDs[0], parentClustsIDs[1]]
            for p in parentClustsIDs:
                if p in trgt2Ctrls:
                    trgt2Ctrls[clusterID] = trgt2Ctrls[p].copy() + trgt2Ctrls[clusterID].copy()
                else:
                    continue

        # 3)- at least one of the two parent clusters identifiers is in the validClustsIDs list
        # identify the already validated parent, and extract only the samples from the
        # non-validated parent cluster.
        elif (parentClustsIDs[0] in validClustsIDs) or (parentClustsIDs[1] in validClustsIDs):
            for p in parentClustsIDs:
                if (p in validClustsIDs):
                    inverseParent = parentClustsIDs[len(parentClustsIDs) - 1 - parentClustsIDs.index(p)]
                    validClustsIDs.append(clusterID)
                    validClustsSamps.append(clust2Samps[inverseParent].copy())
                    checkClusteredSamps[clust2Samps[inverseParent]] = 1
                    trgt2Ctrls[clusterID] = [p]
                    if p in trgt2Ctrls:
                        trgt2Ctrls[clusterID] = trgt2Ctrls[p].copy() + trgt2Ctrls[clusterID].copy()
                else:
                    continue
        clusters = formatClustRes(validClustsIDs, validClustsSamps, trgt2Ctrls, checkClusteredSamps)
    return(clusters)


###########################
# formatClustRes
# formatting the clustering results
# generalisation of cluster identifiers (from 0 to nbclusters-1)
# identifications of samples that could not be clustered placed in the
# list of lists output end as "Samps_ClustFailed"
#
# Args:
# - validClustsIDs (list[int]): clusterIDs from linksMatrix row ordering
# - validClustsSamps (list of list[int]): samples indexes for each clustersID in validClustsIDs
# - trgt2Ctrls (dict([int]:list[int])): a dictionary mapping target cluster IDs to lists of
# control cluster IDs.
# - checkClusteredSamps (np.ndarray[bool]): for each sample index score 0:failed clustering,
# 1:successful clustering
# return a list of list of dim = nbClust * ["newClustID","SampsInCluster","controlsClustersID"]
def formatClustRes(validClustsIDs, validClustsSamps, trgt2Ctrls, checkClusteredSamps):
    # To Fill and returns
    clusters = []

    for i in range(len(validClustsIDs)):
        ctrlList = []  # list of new control clustID
        # case of controlled target cluster replacement of clustID of controls
        if validClustsIDs[i] in trgt2Ctrls:
            for ctrl in trgt2Ctrls[validClustsIDs[i]]:
                ctrlList.append(validClustsIDs.index(ctrl))
        listToFill = [i, validClustsSamps[i], ctrlList]
        clusters.append(listToFill)

    # samples are either included in clusters with small numbers or
    # have distances too far to be included in a cluster
    sampsClustFailed = [i for i, x in enumerate(checkClusteredSamps) if not x]
    clusters.append(["Samps_ClustFailed", sampsClustFailed, ""])
    if len(sampsClustFailed) > 0:
        logger.warn("%s/%s samples fail clustering (see dendogramm plot).", len(sampsClustFailed), len(checkClusteredSamps))

    return(clusters)
