import sys
import os
import numpy as np
import logging
import matplotlib.pyplot


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
# STDZandCheckRes:
#
#
#
#
#
# Args:
#
#
# Returns:
# - clustsResList

def STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps, nogender, clust2SampsGono=None, trgt2CtrlsGono=None, kmeans=None, sexePred=None):
    vSOIs=[]
    SOIs_QCFailed=[]
    for i in range(len(SOIs)):
        if i not in sampsQCfailed:
            vSOIs.append(SOIs[i])
        else:
            SOIs_QCFailed.append(SOIs[i])

    clustsResList=[]
    clustIDList= list(clust2Samps.keys())
    
    for i in range(len(clustIDList)):
        listOflist=[""]*5
        listOflist[0]= i
        listOflist[1]= ", ".join([vSOIs[i] for i in clust2Samps[clustIDList[i]]])
        if clustIDList[i] in trgt2Ctrls.keys():
            listOflist[2]=", ".join(str(clustIDList.index(i)) for i in trgt2Ctrls[clustIDList[i]])

        if len(clust2Samps[clustIDList[i]]) < minSamps and listOflist[2]=="":
            listOflist[3]= 0
            logger.error("cluster N°%s : does not contain enough sample to be valid (%s)", i, len(clust2Samps[clustIDList[i]]) )
        else:
            listOflist[3]= 1

        if nogender:
            listOflist[4]="W"
        else:
            listOflist[4]="A"

        clustsResList.append(listOflist)

    if not nogender:
        clustIDGonoList= list(clust2SampsGono.keys())
        
        for i in range(len(clustIDGonoList)):
            listOflist=[""]*5
            listOflist[0]= len(clustIDList) + i
            listOflist[1]= ", ".join([vSOIs[i] for i in clust2SampsGono[clustIDGonoList[i]]])
            if clustIDGonoList[i] in trgt2CtrlsGono.keys():
                listOflist[2]=", ".join(str(clustIDGonoList.index(i)+ len(clustIDList)) for i in trgt2CtrlsGono[clustIDGonoList[i]])
            if len(clust2SampsGono[clustIDGonoList[i]]) < minSamps and listOflist[2]=="":
                listOflist[3]= 0
                logger.error("cluster N°%s : does not contain enough sample to be valid (%s)", i, len(clust2SampsGono[clustIDGonoList[i]]) )
            else:
                listOflist[3]= 1

            # status détermination
            status= list(set([kmeans[i] for i in clust2SampsGono[clustIDGonoList[i]]]))
            if len(status) == 1:
                listOflist[4]="G_"+sexePred[status[0]]
            else:
                listOflist[4]="G_B"
            
            clustsResList.append(listOflist)

    clustsResList.append(["Samps_QCFailed",", ".join(SOIs_QCFailed),"","",""])
    
    return(clustsResList)


#############################
# printClustersFile:
#
# Args:
# - clustsResList (list of lists[str]): returned by STDZandCheck Function, ie each cluster is a lists
#     of 5 scalars containing [clusterID,Samples,controlledBy,validCluster,status]
#
# Print this data to stdout as a 'clustsFile' (same format parsed by extractClustsFromPrev).
def printClustersFile(clustsResList):
    toPrint = "clusterID\tSamples\tcontrolledBy\tvalidCluster\tstatus"
    print(toPrint)
    for i in range(len(clustsResList)):
        toPrint= "{}\t{}\t{}\t{}\t{}".format(clustsResList[i][0], clustsResList[i][1], clustsResList[i][2], clustsResList[i][3], clustsResList[i][4])
        print(toPrint)


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
    labelArray = np.empty([len(linksMatrix)+1, maxClust + 1], dtype="U1")
    labelArray.fill(" ")
    # labelsGp (list[str]): labels for each sample list to be passed when plotting the dendogram
    labelsGp = []

    keysList=list(clust2Samps.keys())

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
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="png", bbox_inches='tight')
