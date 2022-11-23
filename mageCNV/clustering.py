import sys
import os
import numpy as np
import logging
import matplotlib.pyplot

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
# Groups the samples according to their distance obtained with the hierarchical clustering.
# Several conditions for forming clusters:
# -exceeding the minimum distance threshold ('minDist')
# -below the allowed distance ('maxDist')
# -number of samples greater than minSamps
# Decremental correlation method, if a single sample is grouped with a control cluster it's possible to 
# form a new cluster.
# Args:
#  - FPMarray: is a float numpy array, dim = NbExons x NbSOIs 
#  - SOIs: is the str sample of interest name list 
#  - minDist: is a float variable (1-|r|), it's the minimal distance tolerated to start building clusters 
#  - maxDist: is a float variable, it's the maximal distance to concidered 
#  - minSamps: is an int variable, defining the minimal sample number to validate a cluster
#  - figure: is a boolean: "True" or "False" to generate a figure
#  - outputFile: is a full path (+ file name) for saving a dendogram
# Returns a 2D numpy array, dim= NbSOIs*4 columns[SOIs[str],clusterID[int],controlledBy[str],validitySamps[int]] 

def clustersBuilds(FPMarray, SOIs, minDist, maxDist, minSamps, figure, outputFile):
    ###################################
    # part 1: Calcul distance between samples and apply hierachical clustering
    ###################################
    # Euclidean distance (classical method) not used
    # Absolute correlation distance is unlikely to be a sensible distance when 
    # clustering samples. ( 1-|r| where r is the Pearson correlation)
    correlation = np.round(np.corrcoef(FPMarray,rowvar=False),2)
    dissimilarity = 1 - abs(correlation) 

    # average linkage the best choice when there are different-sized groups
    # "squareform" transform squared distance matrix in a triangular matrix
    # "optimal_ordering": linkage matrix will be reordered so that the distance between
    # successive leaves is minimal.
    # linksMatrix: is a float numpy.ndarray of the hierarchical clustering encoded 
    # as a linkage matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
    linksMatrix = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'average',optimal_ordering=True)

    ###################################
    # part 2: linksMatrix analysis, clusters formation and identification of controls/targets clusters
    ###################################
    ############
    # To Fill
    # clusters: an int 1D numpy array, clusterID associated to SOIsIndex
    clusters=np.zeros(FPMarray.shape[1], dtype=np.int)
    # trgt2Ctrls: target and controls clusters correspondance,
    # key = target clusterID [int], value = list of controls clusterID [int list]
    trgt2Ctrls={}  
    # links2Clusters: cluster formed thanks to the linksMatrix parsing associated with SOIs.
    # key = current clusterID [int], value = list of SOIs indexes [int list] 
    links2Clusters={}

    ############
    # To increment
    # identifies clusters
    # e.g first formed samples cluster => NbRow from linksMatrix +1 (e.q NbSOIs) 
    clusterID=len(linksMatrix)

    ##########################################
    # Main loop: populate clusters, trgt2Ctrls and links2Cluster from linkage matrix
    for clusterLine in linksMatrix:
        clusterID += 1    

        # keep parent clusters ID and convert it to int for extract easily SOIs indexes
        parentClustsIDs = [np.int(clusterLine[0]),np.int(clusterLine[1])]

        distValue = clusterLine[2]
        NbSOIsInClust = clusterLine[3]

        # SOIsIndexInParents: an int list of parent clusters SOIs indexes 
        # nbSOIsInParents: an int list of samples number in each parent clusters
        SOIsIndexInParents,nbSOIsInParents = getParentsClustsInfosPrivate(parentClustsIDs, links2Clusters, len(linksMatrix))

        ################ 
        # CONTROL that the sample number calculation is correct
        # TO REMOVE 
        if (len(SOIsIndexInParents)!=NbSOIsInClust):
            break

        ################
        # populate links2Clusters 
        links2Clusters[clusterID] = SOIsIndexInParents
        
        ##########################
        # CONDITIONS FOR CLUSTER CREATION
        # Populate np.array clusters and dictionnary trgt2Ctrls 
        ##########
        # condition 1: group formation from a minimum to a maximum correlation distance
        # replacement/overwriting of old clusterIDs with the new one for SOIs indexes in 
        # np.array "clusters"
        if (distValue<minDist):
            clusters[SOIsIndexInParents] = clusterID
        
        # distance between minDist and maxDist 
        # cluster selection is possible
        elif ((distValue>=minDist) and (distValue<=maxDist)):

            ##########
            # condition 2: estimate the samples number to create a cluster 
            # insufficient samples, we continue to replace the old ClusterId with the new one
            if (len(SOIsIndexInParents)<minSamps):
                clusters[SOIsIndexInParents] = clusterID

            # sufficient samples number
            else:
                ###############
                # Identification of the different cases 
                # Knowing that we are dealing with two parent clusters 
                
                # Case 1: both parent clusters have sufficient numbers to be controls
                # creation of a new target cluster with the current clusterID (trgt2Ctrls key)
                # the controls are the parents clusterID and their previous control clusterID (trgt2Ctrls list value)
                if ((nbSOIsInParents[0]>=minSamps) and (nbSOIsInParents[1]>=minSamps)):
                    trgt2Ctrls[clusterID] = parentClustsIDs
                    for parentID in parentClustsIDs:   
                        if parentID in trgt2Ctrls:
                                trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID]+trgt2Ctrls[parentID]
                
                # Case 2: one parent has a sufficient number of samples not the second parent 
                # creation of a new target cluster with the current clusterID (trgt2Ctrls key)
                # controls list (trgt2Ctrls list value): the control parent clusterID and its own controls
                # overwrite the old clusterID for the SOIs indexes from the none control parent by the new
                # clusterID in np.array "clusters"
                elif max(SOIsIndexInParents)>20:
                    # the parent control index 
                    # index corresponding to nbSOIsInParents and parentClustsIDs (e.g list: [parent1, parent2])
                    indexCtrlParent = np.argmax(nbSOIsInParents)
                    # the parent index with insufficient samples number
                    indexNewParent = np.argmin(nbSOIsInParents)

                    # populate trgt2Ctrl with the current clusterID (trgt2Ctrls key)
                    # set controls clusterId can be retrieved from the control parent (trgt2Ctrls list value)
                    if (parentClustsIDs[indexCtrlParent] in trgt2Ctrls): # parent control with previous controls
                        trgt2Ctrls[clusterID] = trgt2Ctrls[parentClustsIDs[indexCtrlParent]]
                        trgt2Ctrls[clusterID] = trgt2Ctrls[clusterID]+[parentClustsIDs[indexCtrlParent]]
                    else:# parent control without previous controls
                        trgt2Ctrls[clusterID] = [parentClustsIDs[indexCtrlParent]]

                    # populate "clusters" for SOIs index from the parent with few sample
                    if (indexNewParent==0):
                        clusters[SOIsIndexInParents[:nbSOIsInParents[indexNewParent]]] = clusterID
                    else:
                        clusters[SOIsIndexInParents[-nbSOIsInParents[indexNewParent]:]] = clusterID
                
                # Case 3: each parent cluster has an insufficient number of samples.
                # the current clusterID becomes a cluster control.
                # replacement of all indexed SOIs with the current clusterID for the np.array "clusters"
                else:
                    clusters[SOIsIndexInParents] = clusterID     
        
        # dist>0.25 we stop the loop of linksMatrix
        else:
            break
    ###################################
    # part 3: Standardisation clusterID and create informative lists for populate final np.array 
    ###################################
    clusters, ctrls, validityStatus = STDZAndCheckResPrivate(clusters, trgt2Ctrls, minSamps)

    ###################################
    # part 4: Optionnal plot of a dendogram based on clustering results
    ###################################
    if figure:
        DendogramsPrivate(clusters, ctrls, linksMatrix, minDist, outputFile)
    
    ##########
    return(np.column_stack((SOIs, clusters, ctrls, validityStatus)))


###############################################################################
# printClustersFile:
# print the different types of outputs expected
# Args:
#   - resClustering: is a list of list, dim=NbSOIs*4columns 
#       [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int]
#       can be derived from clustering on all chromosomes or on the autosomes
#   - outFolder: is a str variable, it's the results folder path  
#   - resClusteringGonosomes: is a list of list, dim=NbSOIs*5columns
#       [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int], genderPreds[str]]
#       this argument is optional
# Print this data to stdout as a 'clustersFile'
def printClustersAGFile(resClustering, resClusteringGonosomes, outFolder):
    # 8 columns expected
    clusterFile=open(os.path.join(outFolder,"ResClustering_AutosomesAndGonosomes_"+str(len(resClustering))+"samples.tsv"),'w')
    sys.stdout = clusterFile
    toPrint = "samplesID\tclusterID_A\tcontrolledBy_A\tvaliditySamps_A\tgenderPreds\tclusterID_G\tcontrolledBy_G\tvaliditySamps_G"
    print(toPrint)
    for i in range(len(resClustering)):
        # SOIsID + clusterInfo for autosomes and gonosomes
        toPrint = resClustering[i][0]+"\t"+resClustering[i][1]+"\t"+resClustering[i][2]+\
            "\t"+resClustering[i][3]+"\t"+resClusteringGonosomes[i][4]+"\t"+resClusteringGonosomes[i][1]+\
                "\t"+resClusteringGonosomes[i][2]+"\t"+resClusteringGonosomes[i][3]
        print(toPrint)
    sys.stdout = sys.__stdout__
    clusterFile.close()

###############################################################################
# printClustersFile:
# print the different types of outputs expected
# Args:
#   - resClustering: is a list of list, dim=NbSOIs*4columns 
#       [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int]
#       can be derived from clustering on all chromosomes or on the autosomes
#   - outFolder: is a str variable, it's the results folder path  
#   - resClusteringGonosomes: is a list of list, dim=NbSOIs*5columns
#       [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int], genderPreds[str]]
#       this argument is optional
# Print this data to stdout as a 'clustersFile'
def printClustersFile(resClustering, outFolder):
    # 4 columns expected
    clusterFile=open(os.path.join(outFolder,"ResClustering_"+str(len(resClustering))+"samples.tsv"),'w')
    sys.stdout = clusterFile
    toPrint = "samplesID\tclusterID\tcontrolledBy\tvaliditySamps"
    print(toPrint)
    for i in range(len(resClustering)):
        # SOIsID + clusterInfo 
        toPrint = resClustering[i][0]+"\t"+resClustering[i][1]+"\t"+resClustering[i][2]+"\t"+resClustering[i][3]
        print(toPrint)
    sys.stdout = sys.__stdout__
    clusterFile.close()

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###############################################################################
# getParentsClustsInfosPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Extract parents informations : SOIs indexes list and sample number 
# Arg:
#  - parentClustsIDs: a list containing the two cluster identifiers to be combined
#  - links2Clusters: cluster formed thanks to the linksMatrix parsing associated with SOIs.
#    key = current clusterID, value = list of SOIs indexes 
#  - NbLinks: an int variable, links number in linksMatrix (row count)
# Returns a tuple (SOIsIndexInParents, nbSOIsInParents), each is created here:
#  - SOIsIndexInParents: an int list of parent clusters SOIs indexes 
#  - nbSOIsInParents: an int list of samples number in each parent clusters

def getParentsClustsInfosPrivate(parentClustsIDs, links2Clusters, NbLinks):
    SOIsIndexInParents = []
    nbSOIsInParents = []

    for parentID in parentClustsIDs:
        #####
        # where it's a sample identifier not a cluster
        # the clusterID corresponds to the SOI index
        if (parentID<=NbLinks):
            SOIsIndexInParents.append(parentID)
            nbSOIsInParents.append(1)
        #####
        # where it's a cluster identifier
        # we get indexes lists
        else:
            SOIsIndexInParents = SOIsIndexInParents+links2Clusters[parentID]
            nbSOIsInParents.append(len(links2Clusters[parentID])) 
    return(SOIsIndexInParents, nbSOIsInParents)

###############################################################################
# STDZAndCheckResPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Standardization: replacement of the clusterIDs deduced from linksMatrix by identifiers ranging from 0
# to the total number of clusters.
# the new identifiers are assigned according to the decreasing correlation level.
# Checking: the samples are in a cluster with a sufficient size. 
# if not returns a warning message and changes the validity status. 
# Necessary for the calling step.
# Args:
#  - clusters: an int numpy array containing clusterID from linksMatrix for each sample
#  - trgt2Ctrls: a dictionary for target cluster and controls clusters correspondance,
#    key = target clusterID, value = list of controls clusterID 
#  - minSamps: an int variable, defining the minimal sample number to validate a cluster
# Returns a tuple (clusters, ctrls, validityStatus), only ctrls and validityStatus are created here:
#   - clusters: an int numpy array containing standardized clusterID for each sample
#   - ctrls: a str list containing controls clusterID delimited by "," for each sample 
#   - validityStatus: a boolean numpy array containing the validity status for each sample (1: valid, 0: invalid)

def STDZAndCheckResPrivate(clusters, trgt2Ctrls, minSamps):
    # extraction of two int numpy array
    # uniqueClusterIDs: contains all clusterIDs
    # countsSampsinCluster: contains all sample counts per clusterIDs 
    uniqueClusterIDs, countsSampsinCluster = np.unique(clusters, return_counts=True)

    ##########
    # To Fill
    ctrls = [""]*len(clusters)
    validityStatus = np.ones(len(clusters), dtype=np.int)

    ##########
    # browse all unique cluster identifiers 
    for newClusterID in range(len(uniqueClusterIDs)):
        clusterID = uniqueClusterIDs[newClusterID]
        # selection of sample indices associated with the old clusterID
        Sindex=[i for i in range(len(clusters)) if clusters[i] == clusterID]
        # replacement by the new
        clusters[Sindex] = newClusterID   

        # filling ctrls by replacing clusterIDs with new ones
        if (clusterID in trgt2Ctrls):
            emptylist = []
            for i in trgt2Ctrls[clusterID]:
                if (i in uniqueClusterIDs):
                    emptylist.append(np.where(uniqueClusterIDs==i)[0][0])
            emptylist = ",".join(map(str,emptylist)) 
            for index in Sindex:
                ctrls[index] = emptylist

        # checking the validity: 
        else:
            #the sample(s) were not clustered
            if clusterID==newClusterID:
                logger.warning("%s sample(s) were not clustered (maxDist to be reviewed).",len(Sindex))
                validityStatus[Sindex]=0
            # cluster samples number is not sufficient to establish a correct copies numbers call
            elif (countsSampsinCluster[newClusterID]<minSamps):
                logger.warning("Cluster n°%s has an insufficient samples number = %s ",newClusterID, countsSampsinCluster[newClusterID])
                validityStatus[Sindex]=0

    return(clusters, ctrls, validityStatus)

###############################################################################
# DendogramsPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# visualisation of clustering results
# Args:
# -clusters: an int numpy array containing standardized clusterID for each sample
# -ctrls: a str list containing controls clusterID delimited by "," for each sample 
# -linksMatrix: is a float numpy.ndarray of the hierarchical clustering encoded 
#    as a linkage matrix, dim = (NbSOIs-1)*[clusterID1,clusterID2,distValue,NbSOIsInClust]
# -minDist: is a float variable, sets a minimum threshold 1-|r| for the formation of the first clusters
# -outputFile : full path to save the png 
# Returns a png file in the output folder
def DendogramsPrivate(clusters, ctrls, linksMatrix, minDist, outputFile):
    # maxClust: int variable contains total clusters number
    maxClust = max(clusters)

    # To Fill
    # labelArray: a str 2D numpy array, dim=NbSOIs*NbClusters 
    # contains the status for each cluster as a character:
    # " ": sample does not contribute to the cluster
    # "x": sample contributes to the cluster
    # "-": sample controls the cluster
    labelArray = np.empty([len(clusters), maxClust+1], dtype="U1")
    labelArray.fill(" ") 
    # labelsGp: a str list containing the labels for each sample
    # list to be passed when plotting the dendogram
    labelsGp=[]

    # browse the different cluster identifiers
    for clusterID in range(maxClust+1):
        # retrieving the SOIs involved for the clusterID
        SOIsindex=[i for i in range(len(clusters)) if clusters[i]==clusterID]
        # associate the label for the samples contributing to the clusterID for the 
        # associated cluster index position
        labelArray[SOIsindex,clusterID]="x"    

        # associate the label for the samples controlling the current clusterID  
        if (ctrls[SOIsindex[0]] !=""):
            listctrl = ctrls[SOIsindex[0]].split(",")
            for ctrl in listctrl:
                CTRLindex = [j for j in range(len(clusters)) if clusters[j]==np.int(ctrl)]
                labelArray[CTRLindex,clusterID]="-"

    # browse the np array of labels to build the str list
    for i in labelArray:
        # separtion of labels for readability
        strToBind="  ".join(i)
        labelsGp.append(strToBind)

    # dendogram plot
    matplotlib.pyplot.figure(figsize=(5,20),facecolor="white")
    matplotlib.pyplot.title("Average linkage hierarchical clustering")
    dn1 = scipy.cluster.hierarchy.dendrogram(linksMatrix,orientation='left',
                                            labels=labelsGp, 
                                            color_threshold=minDist)

    matplotlib.pyplot.ylabel("Samples of interest")
    matplotlib.pyplot.xlabel("Absolute Pearson distance (1-|r|)")
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="png", bbox_inches='tight')