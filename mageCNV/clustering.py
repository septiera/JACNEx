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
# returns a pdf in the output folder if the figure option is set.
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
# - clust2Samps (dict(int : list[int])): clusterID associated to SOIsIndex
#   key = clusterID, value = list of SOIsIndex
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
def clustersBuilds(FPMarray, maxCorr, minCorr, minSamps, outputFile=None):
    # - minDist (float): is the distance to start cluster construction
    minDist = (1 - maxCorr)**0.5
    # - maxDist (float): is the distance to finalise the cluster construction
    maxDist = (1 - minCorr)**0.5

    # - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
    #  matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
    linksMatrix = computeSampLinksPrivate(FPMarray)

    (clust2Samps, trgt2Ctrls) = links2ClustersPrivate(linksMatrix, minDist, maxDist, minSamps)

    # Optionnal plot a dendogram based on clustering results
    if outputFile:
        DendogramsPrivate(clust2Samps, trgt2Ctrls, linksMatrix, minDist, outputFile)

    return(clust2Samps, trgt2Ctrls)


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


    return (clusters, ctrls, validSampClust, genderPred)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################
# computeSampLinksPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Pearson correlation distance (sqrt(1-r)) is unlikely to be a sensible
# distance when clustering samples.
# (sqrt(1-r)) is a true distance respecting symmetry, separation and triangular
# inequality
# average linkage method is the best choice when there are different-sized groups
#
# Args:
# - FPMarray (np.ndarray[float]): normalised fragment counts for QC validated
#  samples (VSOIs), dim = NbCoveredExons x NbSOIsQCValidated
#
# Return:
# - linksMatrix (np.ndarray[float]): the hierarchical clustering encoded as a linkage
#  matrix, dim = (NbVSOIs-1)*[clusterID1,clusterID2,distValue,NbVSOIsInClust]
def computeSampLinksPrivate(FPMarray):
    
    correlation = np.round(np.corrcoef(FPMarray, rowvar=False), 2)
    dissimilarity = (1 - correlation)**0.5

    # "squareform" transform squared distance matrix in a triangular matrix
    # "optimal_ordering": linkage matrix will be reordered so that the distance between
    # successive leaves is minimal.
    linksMatrix = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'average', optimal_ordering=True)
    return(linksMatrix)


#############################
# links2ClustersPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
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
def links2ClustersPrivate(linksMatrix, minDist, maxDist, minSamps):
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
        
        # VSOIsIndexInParents (list[int]): parent clusters VSOIs indexes
        # nbVSOIsInParents (list[int]): samples number in each parent clusters
        (VSOIsIndexInParents, nbVSOIsInParents) = getParentsClustsInfosPrivate(parentClustsIDs, links2Clusters, len(linksMatrix))

        ################
        # Fill links2Clusters
        links2Clusters[clusterID] = VSOIsIndexInParents
        
        """
        ################
        # DEV CONTROL ; check that the sample number calculation for a cluster is correct
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
        if (distValue < minDist):
            clust2Samps[clusterID] = VSOIsIndexInParents
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
            if (len(VSOIsIndexInParents) >= minSamps):
                
                ###############
                # Different cases to complete the two dictionaries
                # Knowing that we are dealing with two parent clusters
                # Case 1: both parent clusters have sufficient numbers to be controls
                if ((nbVSOIsInParents[0] >= minSamps) and (nbVSOIsInParents[1] >= minSamps)):
                    
                    # parent clusters are previously saved in clust2samps
                    if clust2Samps.keys() >= {parentClustsIDs[0], parentClustsIDs[1]}:
                        # fill trgt2Ctrl
                        trgt2Ctrls[clusterID] = parentClustsIDs
                        # case that parentClustIDs are already controlled
                        for parentID in parentClustsIDs:
                            if parentID in trgt2Ctrls:
                                trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID] + trgt2Ctrls[parentID]
                    # parent clusterID not in clust2samps
                    else:
                        clust2Samps[clusterID] = VSOIsIndexInParents

                # Case 2: one parent has a sufficient number of samples not the second parent
                elif max(VSOIsIndexInParents) >= 20:
                    # identification of the control parent and the target parent
                    # index corresponding to nbVSOIsInParents and parentClustsIDs (e.g list: [parent1, parent2])
                    indexCtrlParent = np.argmax(nbVSOIsInParents) # control
                    indexNewParent = np.argmin(nbVSOIsInParents) # target

                    
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
                            clust2Samps[clusterID] = VSOIsIndexInParents[:nbVSOIsInParents[indexNewParent]]
                        else:
                            clust2Samps[clusterID] = VSOIsIndexInParents[-nbVSOIsInParents[indexNewParent]:]
                    else:
                        clust2Samps[clusterID] = VSOIsIndexInParents
                     
                    if parentClustsIDs[indexNewParent] in clust2Samps:  
                        del clust2Samps[parentClustsIDs[indexNewParent]]
                    
                # Case 3: each parent cluster has an insufficient number of samples.
                # the current clusterID becomes a control cluster.
                else:
                    clust2Samps[clusterID] = VSOIsIndexInParents
                    if parentClustsIDs[0] in clust2Samps:  
                        del clust2Samps[parentClustsIDs[0]]
                    if parentClustsIDs[1] in clust2Samps:  
                        del clust2Samps[parentClustsIDs[1]]

            # clusters too small 
            # complete clust2Samps with new clusterID and remove old clustersID
            else:
                clust2Samps[clusterID] = VSOIsIndexInParents
                for key in parentClustsIDs:
                    if key in clust2Samps:
                        del clust2Samps[key]
            
        # current distance larger than maxDist we stop the loop on linksMatrix rows
        elif (distValue > maxDist):
            break
    return(clust2Samps, trgt2Ctrls)


#############################
# getParentsClustsInfosPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# function used in another private function links2ClustersFormationPrivate
# Extract parents informations : VSOIs indexes list and samples number
#
# Args:
# - parentClustsIDs (list[int]): two parents clusters identifiers to be combined
# - links2Clusters (dict(int : list[int])): cluster formed thanks to the linksMatrix
#   parsing associated with VSOIs.
#   key = current clusterID, value = list of VSOIs indexes
# - nbLinks (int): links number in linksMatrix (row count)
#
# Returns a tuple (VSOIsIndexInParents, nbVSOIsInParents), each is created here:
# - VSOIsIndexInParents (list[int]): parent clusters VSOIs indexes
# - nbVSOIsInParents (list[int]): samples number in each parent clusters
def getParentsClustsInfosPrivate(parentClustsIDs, links2Clusters, NbLinks):
    VSOIsIndexInParents = []
    nbVSOIsInParents = []

    for parentID in parentClustsIDs:
        #####
        # where it's a sample identifier not a cluster
        # the clusterID corresponds to the SOI index
        if (parentID <= NbLinks):
            VSOIsIndexInParents.append(parentID)
            nbVSOIsInParents.append(1)
        #####
        # where it's a cluster identifier
        # we get indexes lists
        else:
            VSOIsIndexInParents = VSOIsIndexInParents + links2Clusters[parentID]
            nbVSOIsInParents.append(len(links2Clusters[parentID]))
    return(VSOIsIndexInParents, nbVSOIsInParents)

#############################
# DendogramsPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# visualisation of clustering results
# Args:
# - clust2Samps (dict(int : list[int])): clusterID associated to valid sample indexes
#   key = clusterID, value = list of valid sample indexes
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
# - linksMatrix (np.ndarray[float])
# - minDist (float): is the distance to start cluster construction
# - outputFile (str): full path to save the png
# Returns a png file in the output folder
def DendogramsPrivate(clust2Samps, trgt2Ctrls, linksMatrix, minDist, outputFile):
    # maxClust: int variable contains total clusters number
    maxClust = len(clust2Samps.keys())

    # To Fill
    # labelArray (np.ndarray[str]): status for each cluster as a character, dim=NbSOIs*NbClusters
    # " ": sample does not contribute to the cluster
    # "x": sample contributes to the cluster
    # "-": sample controls the cluster
    labelArray = np.empty([len(linksMatrix+1), maxClust + 1], dtype="U1")
    labelArray.fill(" ")
    # labelsGp (list[str]): labels for each sample list to be passed when plotting the dendogram
    labelsGp = []
    
    keysList=clust2Samps.keys()

    # browse the different cluster identifiers
    for clusterID in range(len(keysList)):
        # retrieving the SOIs involved for the clusterID
        SOIsindex = clust2Samps[keysList[clusterID]]
        # associate the label for the samples contributing to the clusterID for the
        # associated cluster index position
        labelArray[SOIsindex, clusterID] = "x"

        # associate the label for the samples controlling the current clusterID
        if keysList[clusterID] in trgt2Ctrls.keys():
            listctrl = trgt2Ctrls[keysList[clusterID]]
            for ctrl in listctrl:
                CTRLindex = clust2Samps[ctrl]
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
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="pdf", bbox_inches='tight')
