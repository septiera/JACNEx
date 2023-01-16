import sys
import os
import numpy as np
import logging
import matplotlib.pyplot
# import sklearn submodule for Kmeans calculation
import sklearn.cluster

import mageCNV.genderDiscrimination

# different scipy submodules are used for the application of hierachical clustering
import scipy.cluster.hierarchy
import scipy.spatial.distance

# prevent matplotlib DEBUG messages filling the logs when we are in DEBUG loglevel
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################
# clustersBuilds
# Groups the QC validated samples according to their distance obtained with a
# hierarchical clustering.
#
# Args:
#  - FPMarray (np.ndarray[float]): normalised fragment counts for QC validated samples,
#  dim = NbCoveredExons x NbSOIsQCValidated
#  - maxCorr (float): maximal Pearson correlation score tolerated by the user to start
#   build clusters
#  - minCorr (float): minimal Pearson correlation score tolerated by the user to end
#   build clusters
#  - minSamps (int): minimal sample number to validate a cluster
#  - outputFile (str): full path (+ file name) for saving a dendogram
#
# Returns a tuple (clusters, ctrls, validSampClust):
#  - clusters (np.ndarray[int]): clusterID for each sample
#  - ctrls (list[str]): controls clusterID delimited by "," for each sample
#  - validSampClust (np.ndarray[int]): validity status for each sample passed
#   quality control (1: valid, 0: invalid), dim = NbSOIs
def clustersBuilds(FPMarray, maxCorr, minCorr, minSamps, outputFile=None):
    # compute distance thresholds for cluster construction
    # depends on user-defined correlation levels
    # - minDist (float): is the distance to start cluster construction
    minDist = (1 - maxCorr)**0.5
    # - maxDist (float): is the distance to finalise the cluster construction
    maxDist = (1 - minCorr)**0.5

    # Compute distances between samples and apply hierachical clustering
    # - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
    #  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
    linksMatrix = computeSampLinksPrivate(FPMarray)

    # linksMatrix analysis, clusters formation and identification of controls/targets clusters
    # - clusters (np.ndarray[int]): clusterID associated to SOIsIndex
    # - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
    #   key = target clusterID, value = list of controls clusterID
    (clusters, trgt2Ctrls) = links2ClustersFormationPrivate(FPMarray.shape[1], linksMatrix, minDist, maxDist, minSamps)

    # Standardisation clusterID and create informative lists for populate final np.array
    # - clusters (np.ndarray[int]): clusterID associated to SOIsIndex
    # - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
    #   key = target clusterID, value = list of controls clusterID
    (clusters, ctrls, validSampClust) = STDZAndCheckPrivate(clusters, trgt2Ctrls, minSamps)

    # Optionnal plot a dendogram based on clustering results
    if outputFile:
        DendogramsPrivate(clusters, ctrls, linksMatrix, minDist, outputFile)

    ##########
    return(clusters, ctrls, validSampClust)


#############################
# printClustersFile:
#
# Args:
# - nogender (boolean): Sex discrimination for clustering (default: False)
# - SOIs (list[str]): sampleIDs
# - validSampQC (np.ndarray(boolean)): validity status for each SOIs index after
# quality control (1: valid, 0: invalid)
# - validSampClust (np.ndarray[boolean]): validity status for each SOIs
# validated by quality control after clustering (1: valid, 0: invalid)
# - clusters (np.ndarray[int]): clusterID for each sample
# - ctrls (list[str]): controls clusterID delimited by "," for each sample
# - outFolder (str): output folder path
# - clustersG, ctrlsG, validSampsClustG [optionnal]: same as previous but for gonosomes
# - genderPred (list[str])[optionnal]:  gender (e.g "M" or "F"), for each sample
#
# Results are printed to stdout folder:
# - a TSV file format, describe the clustering results,
# Case the user doesn't want to discriminate genders dim = NbSOIs*5 columns:
#    1) "sampleID" (str): name of interest samples,
#    2) "validQC" (int): specifying if a sample pass quality control (valid=1) or not (dubious=0)
#    3) "validCluster" (int): specifying if a sample is valid (1) or dubious doesn't pass
#                                QC step (0) or doesn't pass clustering validation (0).
#    4) "clusterID" (int): clusters identifiers obtained through the normalized fragment counts.
#    5) "controlledBy" (str): clusters identifiers controlling the sample cluster, a
#                               comma-separated string of int values (e.g "1,2").
#                               If not controlled empty string.
# Case the user want discriminate genders dim = NbSOIs*9 columns:
# The columns 3-5 are from the analysis of autosome coverage profiles (A)
#    6) "genderPreds" (str): "M"=Male or "F"=Female deduced by kmeans,
# The columns 7-9 are the same as 3-5 but are from the analysis of gonosome coverage profiles (G)
def printClustersFile(nogender, SOIs, validSampQC, validSampClust, clusters, ctrls, outFolder, validSampsClustG=False, clustersG=False, ctrlsG=False, genderPred=False):

    if nogender:
        # 5 columns expected
        file_name = "ResClustering_{}samples.tsv".format(len(SOIs))
        cluster_file = open(os.path.join(outFolder, file_name), 'w')
        to_print = "samplesID\tvalidQC\tvalidCluster\tclusterID\tcontrolledBy"
        cluster_file.write(to_print + '\n')
        counter = 0
        for i in range(len(SOIs)):
            if validSampQC[i] != 0:
                to_print = "{}\t{}\t{}\t{}\t{}".format(SOIs[i], validSampQC[i], validSampClust[counter], clusters[counter], ctrls[counter])
                counter += 1
            else:
                to_print = "{}\t{}\t{}\t{}\t{}".format(SOIs[i], validSampQC[i], 0, "", "")
            cluster_file.write(to_print + '\n')
        cluster_file.close()

    else:
        # 9 columns expected
        file_name = "ResClustering_AutosomesAndGonosomes_{}samples.tsv".format(len(SOIs))
        cluster_file = open(os.path.join(outFolder, file_name), 'w')
        to_print = "samplesID\tvalidQC\tvalidCluster_A\tclusterID_A\tcontrolledBy_A\tgenderPreds\tvalidCluster_G\tclusterID_G\tcontrolledBy_G"
        cluster_file.write(to_print + '\n')
        counter = 0
        for i in range(len(SOIs)):
            if validSampQC[i] != 0:
                # SOIsID + clusterInfo for autosomes and gonosomes
                to_print = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(SOIs[i], validSampQC[i], validSampClust[counter], 
                                                                       clusters[counter], ctrls[counter], genderPred[counter],
                                                                       validSampsClustG[counter], clustersG[counter], ctrlsG[counter])
                counter += 1
            else:
                to_print = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(SOIs[i], validSampQC[i], 0, "", "", "", 0, "", "")
            cluster_file.write(to_print + '\n')
        cluster_file.close()


#############################
# GonosomesClustersBuilds:
# - Performs a kmeans on the coverages of the exons present in the gonosomes,
# to identify two groups normally associated with the gender.
# - Assigns gender to each group of Kmeans based on their coverage ratios.
# - Performs two clustering analysis
# - aggregates the results for all valid samples.
# Warning: the male and female groups IDs all start with 0, so it is necessary
# to refer to the genderPred list/column to avoid integrating males with females.
#
# Args:
# - genderInfo (list of list[str]):contains informations for the gender
# identification, ie ["gender identifier","specific chromosome"].
# - validCounts (np.ndarray[float]): normalised fragment counts for QC validated samples,
#  dim = NbCoveredExons x NbSOIsQCValidated
# - gonoIndex (dict(str: list(int))): key=GonosomeID(e.g 'chrX'),
# value=list of gonosome exon index.
# - maxCorr (float): maximal Pearson correlation score tolerated by the user to start
#   build clusters
# - minCorr (float): minimal Pearson correlation score tolerated by the user to end
#   build clusters
# - minSamps (int): minimal sample number to validate a cluster
# - figure (boolean): "True" => produce a figure
# - outFolder (str): output folder path

#
# Returns a tuple (clusters, ctrls, validSampClust, genderPred), all objects are created here:
# - clusters (np.ndarray[int]): clusterID for each sample
# - ctrls (list[str]): controls clusterID delimited by "," for each sample
# - validSampClust (np.ndarray[int]): validity status for each sample passed
#   quality control (1: valid, 0: invalid), dim = NbSOIs
# - genderPred (list[str]): genderID delimited for each SOIs (e.g: "M" or "F")
def GonosomesClustersBuilds(genderInfo, validCounts, gonoIndex, maxCorr, minCorr, minSamps, figure, outFolder):

    # cutting normalized count data according to gonosomal exons
    # - gonoIndexFlat (np.ndarray[int]): flat gonosome exon indexes list
    gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist])
    # - gonosomesFPM (np.ndarray[float]): normalized fragment counts for valid samples, exons covered
    # in gonosomes
    gonosomesFPM = np.take(validCounts, gonoIndexFlat, axis=0)

    ### To Fill and returns
    clusters = np.zeros(gonosomesFPM.shape[1], dtype=np.int)
    ctrls = [""] * gonosomesFPM.shape[1]
    validSampClust = np.ones(gonosomesFPM.shape[1], dtype=np.int)
    genderPred = [""] * gonosomesFPM.shape[1]

    # Performs an empirical method (kmeans) to dissociate male and female.
    # consider only the coverage for the exons present in the gonosomes
    # Kmeans with k=2 (always)
    # - kmeans (list[int]): groupID predicted by Kmeans ordered on SOIsIndex
    kmeans = sklearn.cluster.KMeans(n_clusters=len(genderInfo), random_state=0).fit(gonosomesFPM.T).predict(gonosomesFPM.T)

    # compute coverage rate for the different Kmeans groups and on
    # the different gonosomes to associate the Kmeans group with a gender
    # - gender2Kmeans (list[str]): genderID (e.g ["M","F"]), the order
    # correspond to KMeans groupID (gp1=M, gp2=F)
    gender2Kmeans = mageCNV.genderDiscrimination.genderAttribution(kmeans, validCounts, gonoIndex, genderInfo)

    ####################
    # Independent clustering for the two Kmeans groups
    for genderGp in range(len(gender2Kmeans)):
        # - sampsIndexGp (np.ndarray[int]): indexes of samples of interest for a given gender
        sampsIndexGp = np.where(kmeans == genderGp)[0]
        # - gonosomesFPMGp (np.ndarray[float]): extraction of fragment counts data only for samples
        #  of the selected gender
        gonosomesFPMGp = gonosomesFPM[:, sampsIndexGp]

        # if the user wants the dendograms in the output
        if figure:
            outputFile = os.path.join(outFolder, "Dendogram_" + str(len(sampsIndexGp)) + "Samps_gonosomes_" + gender2Kmeans[genderGp] + ".png")
            (tmpClusters, tmpCtrls, tmpValidityStatus) = clustersBuilds(gonosomesFPMGp, maxCorr, minCorr, minSamps, outputFile)
        else:
            (tmpClusters, tmpCtrls, tmpValidityStatus) = clustersBuilds(gonosomesFPMGp, maxCorr, minCorr, minSamps)

        # populate clusterG, ctrlsG, validityStatusG, genderPred
        for index in range(len(sampsIndexGp)):
            clusters[sampsIndexGp[index]] = tmpClusters[index]
            ctrls[sampsIndexGp[index]] = tmpCtrls[index]
            validSampClust[sampsIndexGp[index]] = tmpValidityStatus[index]
            genderPred[sampsIndexGp[index]] = gender2Kmeans[genderGp]

    return (clusters, ctrls, validSampClust, genderPred)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################
# computeSampLinksPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# compute distances between samples and apply hierachical clustering
#
# Args:
# - FPMarray (np.ndarray[float]): normalised fragment counts for QC validated
#  samples, dim = NbCoveredExons x NbSOIsQCValidated
#
# Return:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
def computeSampLinksPrivate(FPMarray):
    # Pearson correlation distance (sqrt(1-r)) is unlikely to be a sensible
    # distance when clustering samples.
    # (sqrt(1-r)) is a true distance respecting symmetry, separation and triangular
    # inequality
    correlation = np.round(np.corrcoef(FPMarray, rowvar=False), 2)
    dissimilarity = (1 - correlation)**0.5

    # average linkage the best choice when there are different-sized groups
    # "squareform" transform squared distance matrix in a triangular matrix
    # "optimal_ordering": linkage matrix will be reordered so that the distance between
    # successive leaves is minimal.
    linksMatrix = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'average', optimal_ordering=True)
    return(linksMatrix)


#############################
# links2ClustersFormationPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# linksMatrix analysis, clusters formation and identification of controls/targets clusters
# Conditions list for building a cluster:
# 1- The distance between samples in the cluster must be within a certain range,
# specified by the minDist and maxDist parameters.
# 2- The number of samples in the cluster must be above a certain threshold,
# specified by the minSamps parameter.
#
# Args:
# - nbSamps2Clust [int]: number of samples analysed
# - linksMatrix (np.ndarray[float])
# - minDist (float): is the distance to start cluster construction
# - maxDist (float): is the distance to stop cluster construction
# - minSamps (int): minimal sample number to validate a cluster
#
# Returns a tupple (clusters, trgt2Ctrls), each is created here:
# - clusters (np.ndarray[int]): clusterID associated to SOIsIndex
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
def links2ClustersFormationPrivate(nbSamps2Clust, linksMatrix, minDist, maxDist, minSamps):
    ############
    # To Fill and returns
    clusters = np.zeros(nbSamps2Clust, dtype=np.int)
    trgt2Ctrls = {}

    # To Fill, not returns
    # - links2Clusters (dict(int : list[int])): cluster formed thanks to the linksMatrix
    #   parsing associated with SOIs.
    #   key = current clusterID, value = list of SOIs indexes
    links2Clusters = {}

    ############
    # To increment
    # identifies clusters
    # e.g first formed samples cluster => NbRow from linksMatrix +1 (e.q NbSOIs)
    clusterID = len(linksMatrix)

    ##########################################
    # Main loop: populate clusters, trgt2Ctrls and links2Cluster from linkage matrix
    for clusterLine in linksMatrix:
        clusterID += 1

        # keep parent clusters ID and convert it to int for extract easily SOIs indexes
        parentClustsIDs = [np.int(clusterLine[0]), np.int(clusterLine[1])]

        distValue = clusterLine[2]
        NbSOIsInClust = clusterLine[3]

        # SOIsIndexInParents (list[int]): parent clusters SOIs indexes
        # nbSOIsInParents (list[int]): samples number in each parent clusters
        (SOIsIndexInParents, nbSOIsInParents) = getParentsClustsInfosPrivate(parentClustsIDs, links2Clusters, len(linksMatrix))

        ################
        # CONTROL that the sample number calculation is correct
        # TO REMOVE
        if (len(SOIsIndexInParents) != NbSOIsInClust):
            break

        ################
        # populate links2Clusters
        links2Clusters[clusterID] = SOIsIndexInParents

        ##########################
        # CONDITIONS FOR CLUSTER CREATION
        # Populate "clusters" and "trgt2Ctrls"
        ##########
        # condition 1: group formation from a minimum to a maximum correlation distance
        # replacement/overwriting of old clusterIDs with the new one for SOIs indexes in
        # np.array "clusters"
        if (distValue < minDist):
            clusters[SOIsIndexInParents] = clusterID

        # current distance is between minDist and maxDist
        # cluster selection is possible
        elif ((distValue >= minDist) and (distValue <= maxDist)):

            ##########
            # condition 2: estimate the samples number to create a cluster
            # insufficient samples, we continue to replace the old ClusterId with the new one
            if (len(SOIsIndexInParents) < minSamps):
                clusters[SOIsIndexInParents] = clusterID

            # sufficient samples number
            else:
                ###############
                # Identification of the different cases
                # Knowing that we are dealing with two parent clusters

                # Case 1: both parent clusters have sufficient numbers to be controls
                # creation of a new target cluster with the current clusterID (trgt2Ctrls key)
                # the controls are the parents clusterID and their previous control clusterID (trgt2Ctrls list value)
                if ((nbSOIsInParents[0] >= minSamps) and (nbSOIsInParents[1] >= minSamps)):
                    trgt2Ctrls[clusterID] = parentClustsIDs
                    for parentID in parentClustsIDs:
                        if parentID in trgt2Ctrls:
                            trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID] + trgt2Ctrls[parentID]

                # Case 2: one parent has a sufficient number of samples not the second parent
                # creation of a new target cluster with the current clusterID (trgt2Ctrls key)
                # controls list (trgt2Ctrls list value): the control parent clusterID and its own controls
                # overwrite the old clusterID for the SOIs indexes from the none control parent by the new
                # clusterID in np.array "clusters"
                elif max(SOIsIndexInParents) > 20:
                    # the parent control index
                    # index corresponding to nbSOIsInParents and parentClustsIDs (e.g list: [parent1, parent2])
                    indexCtrlParent = np.argmax(nbSOIsInParents)
                    # the parent index with insufficient samples number
                    indexNewParent = np.argmin(nbSOIsInParents)

                    # populate trgt2Ctrl with the current clusterID (trgt2Ctrls key)
                    # set controls clusterId can be retrieved from the control parent (trgt2Ctrls list value)
                    if (parentClustsIDs[indexCtrlParent] in trgt2Ctrls):  # parent control with previous controls
                        trgt2Ctrls[clusterID] = trgt2Ctrls[parentClustsIDs[indexCtrlParent]]
                        trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID] + [parentClustsIDs[indexCtrlParent]]
                    else:  # parent control without previous controls
                        trgt2Ctrls[clusterID] = [parentClustsIDs[indexCtrlParent]]

                    # populate "clusters" for SOIs index from the parent with few sample
                    if (indexNewParent == 0):
                        clusters[SOIsIndexInParents[:nbSOIsInParents[indexNewParent]]] = clusterID
                    else:
                        clusters[SOIsIndexInParents[-nbSOIsInParents[indexNewParent]:]] = clusterID

                # Case 3: each parent cluster has an insufficient number of samples.
                # the current clusterID becomes a cluster control.
                # replacement of all indexed SOIs with the current clusterID for the np.array "clusters"
                else:
                    clusters[SOIsIndexInParents] = clusterID

        # current distance larger than maxDist we stop the loop on rows from linksMatrix
        else:
            break
    return(clusters, trgt2Ctrls)


#############################
# getParentsClustsInfosPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# function used in another private function links2ClustersFormationPrivate
# Extract parents informations : SOIs indexes list and sample number
#
# Args:
# - parentClustsIDs (list[int]): two parents clusters identifiers to be combined
# - links2Clusters (dict(int : list[int])): cluster formed thanks to the linksMatrix
#   parsing associated with SOIs.
#   key = current clusterID, value = list of SOIs indexes
# - nbLinks (int): links number in linksMatrix (row count)
#
# Returns a tuple (SOIsIndexInParents, nbSOIsInParents), each is created here:
# - SOIsIndexInParents (list[int]): parent clusters SOIs indexes
# - nbSOIsInParents (list[int]): samples number in each parent clusters
def getParentsClustsInfosPrivate(parentClustsIDs, links2Clusters, NbLinks):
    SOIsIndexInParents = []
    nbSOIsInParents = []

    for parentID in parentClustsIDs:
        #####
        # where it's a sample identifier not a cluster
        # the clusterID corresponds to the SOI index
        if (parentID <= NbLinks):
            SOIsIndexInParents.append(parentID)
            nbSOIsInParents.append(1)
        #####
        # where it's a cluster identifier
        # we get indexes lists
        else:
            SOIsIndexInParents = SOIsIndexInParents + links2Clusters[parentID]
            nbSOIsInParents.append(len(links2Clusters[parentID]))
    return(SOIsIndexInParents, nbSOIsInParents)


#############################
# STDZAndCheckPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Standardization: replacement of the clusterIDs deduced from linksMatrix by identifiers
# ranging from 0 to the total number of clusters.
# the new identifiers are assigned according to the decreasing correlation level.
# Checking: the samples are in a cluster with a sufficient size.
# if not returns a warning message and changes the validity status.
# Necessary for the calling step.
#
# Args:
# - clusters (np.ndarray[int]): clusterID associated to SOIsIndex
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
# - minSamps (int): minimal sample number to validate a cluster
#
# Returns a tuple (clusters, ctrls, validSampClust), only ctrls and validSampClust are created here:
# - clusters (np.ndarray[int]): standardized clusterID for each sample
# - ctrls (list[str]): controls clusterID delimited by "," for each sample
# - validSampClust (np.ndarray[int]): validity status for each sample passed
#   quality control (1: valid, 0: invalid), dim = NbSOIs
def STDZAndCheckPrivate(clusters, trgt2Ctrls, minSamps):
    # - uniqueClusterIDs (np.ndarray[int]): contains all clusterIDs
    # - countsSampsinCluster (np.ndarray[int]): contains all sample counts per clusterIDs
    uniqueClusterIDs, countsSampsinCluster = np.unique(clusters, return_counts=True)

    ##########
    # To Fill
    ctrls = [""] * len(clusters)
    validSampClust = np.ones(len(clusters), dtype=np.int)

    # browse all unique cluster identifiers
    for newClusterID in range(len(uniqueClusterIDs)):
        clusterID = uniqueClusterIDs[newClusterID]
        # selection of sample indexes associated with the old clusterID
        Sindex = [i for i in range(len(clusters)) if clusters[i] == clusterID]
        # replacement by the new
        clusters[Sindex] = newClusterID

        # fill ctrls by replacing clusterIDs with new ones
        if (clusterID in trgt2Ctrls):
            emptylist = []
            for i in trgt2Ctrls[clusterID]:
                if (i in uniqueClusterIDs):
                    emptylist.append(np.where(uniqueClusterIDs == i)[0][0])
            emptylist = ",".join(map(str, emptylist))
            for index in Sindex:
                ctrls[index] = emptylist

        # check the validity:
        else:
            # the sample(s) were not clustered
            if clusterID == newClusterID:
                logger.warning("%s sample(s) were not clustered (maxDist to be reviewed).", len(Sindex))
                validSampClust[Sindex] = 0
            # cluster samples number is not sufficient to establish a correct copies numbers call
            elif (countsSampsinCluster[newClusterID] < minSamps):
                logger.warning("Cluster n°%s has an insufficient samples number = %s ", newClusterID, countsSampsinCluster[newClusterID])
                validSampClust[Sindex] = 0

    return(clusters, ctrls, validSampClust)


#############################
# DendogramsPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# visualisation of clustering results
# Args:
# - clusters (np.ndarray[int]): standardized clusterID for each sample
# - ctrls (list[str]): controls clusterID delimited by "," for each sample
# - linksMatrix (np.ndarray[float])
# - minDist (float): is the distance to start cluster construction
# - outputFile (str): full path to save the png
# Returns a png file in the output folder
def DendogramsPrivate(clusters, ctrls, linksMatrix, minDist, outputFile):
    # maxClust: int variable contains total clusters number
    maxClust = max(clusters)

    # To Fill
    # labelArray (np.ndarray[str]): status for each cluster as a character, dim=NbSOIs*NbClusters
    # " ": sample does not contribute to the cluster
    # "x": sample contributes to the cluster
    # "-": sample controls the cluster
    labelArray = np.empty([len(clusters), maxClust + 1], dtype="U1")
    labelArray.fill(" ")
    # labelsGp (list[str]): labels for each sample list to be passed when plotting the dendogram
    labelsGp = []

    # browse the different cluster identifiers
    for clusterID in range(maxClust + 1):
        # retrieving the SOIs involved for the clusterID
        SOIsindex = [i for i in range(len(clusters)) if clusters[i] == clusterID]
        # associate the label for the samples contributing to the clusterID for the
        # associated cluster index position
        labelArray[SOIsindex, clusterID] = "x"

        # associate the label for the samples controlling the current clusterID
        if (ctrls[SOIsindex[0]] != ""):
            listctrl = ctrls[SOIsindex[0]].split(",")
            for ctrl in listctrl:
                CTRLindex = [j for j in range(len(clusters)) if clusters[j] == np.int(ctrl)]
                labelArray[CTRLindex, clusterID] = "-"

    # browse the np array of labels to build the str list
    for i in labelArray:
        # separation of labels for readability
        strToBind = "  ".join(i)
        labelsGp.append(strToBind)

    # dendogram plot
    matplotlib.pyplot.figure(figsize=(15, 5), facecolor="white")
    matplotlib.pyplot.title("Average linkage hierarchical clustering")
    dn1 = scipy.cluster.hierarchy.dendrogram(linksMatrix, labels=labelsGp, color_threshold=minDist)
    matplotlib.pyplot.ylabel("Distance √(1-ρ) ")
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="png", bbox_inches='tight')
