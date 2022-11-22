import sys
import os
import numpy as np
import logging
import matplotlib.pyplot

# different scipy submodules are used for the application of hierachical clustering 
import scipy.cluster.hierarchy 
import scipy.spatial.distance  

# import sklearn submodule for  Kmeans calculation
import sklearn.cluster

# prevent matplotlib DEBUG messages filling the logs when we are in DEBUG loglevel
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

# prevent numba DEBUG messages filling the logs when we are in DEBUG loglevel
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################


###############################################################################
# getGenderInfos: 
# From a list of exons, identification of gonosomes and genders.
# These gonosomes are predefined and limited to the X,Y,Z,W chromosomes present
# in most species (mammals, birds, fish, reptiles).
# The number of genders is therefore limited to 2, i.e. Male and Female
# Arg:import scipy.cluster.hierarchy
# Returns a tuple (gonoIndexDict, gendersInfos), each are created here:
# -> 'gonoIndexDict' is a dictionary where key=GonosomeID(e.g 'chrX')[str], 
# value=list of gonosome exon index [int]. It's populated from the exons list. 
# -> 'gendersInfos' is a str list of lists, contains informations for the gender
# identification, ie ["gender identifier","particular chromosome"].
# The indexes of the different lists are important:
# index 0: gender carrying a unique gonosome not present in the other gender (e.g. human => M:XY)
# index 1: gender carrying two copies of the same gonosome (e.g. human => F:XX) 

def getGenderInfos(exons):
    # pre-defined list of gonosomes
    # the order of the identifiers is needed to more easily identify the 
    # combinations of chromosomes. 
    # combinations: X with Y and Z with W + alphabetical order
    gonoChromList = ["X", "Y", "W", "Z"]
    # reading the first line of "exons" for add "chr" to 'gonoChromList' in case 
    # of this line have it
    if (exons[0][0].startswith("chr")):
        gonoChromList = ["chr" + letter for letter in gonoChromList]
    
    # for each exon in 'gonoChromList', add exon index in int list value for the 
    # correspondant gonosome identifier (str key). 
    gonoIndexDict=dict()
    for exonIndex in range(len(exons)):
        if (exons[exonIndex][0] in gonoChromList):
            if (exons[exonIndex][0] in gonoIndexDict):
                gonoIndexDict[exons[exonIndex][0]].append(exonIndex)
            # initialization of a key, importance of defining the value as a list 
            # to allow filling with the next indices.
            else:
                gonoIndexDict[exons[exonIndex][0]] = [exonIndex]
        # exon in an autosome
        # no process next
        else:
            continue
            
    # the dictionary keys may not be sorted alphabetically
    # needed to compare with gonoChromList      
    sortKeyGonoList = list(gonoIndexDict.keys())
    sortKeyGonoList.sort()
    if (sortKeyGonoList==gonoChromList[:2]):
        # Human case:
        # index 0 => Male with unique gonosome chrY
        # index 1 => Female with 2 chrX 
        genderInfoList = [["M",sortKeyGonoList[1]],["F",sortKeyGonoList[0]]]
    elif (sortKeyGonoList==gonoChromList[2:]):
        # Reptile case:
        # index 0 => Female with unique gonosome chrW
        # index 1 => Male with 2 chrZ 
        genderInfoList = [["F",sortKeyGonoList[0]],["M",sortKeyGonoList[1]]]
    else:
        logger.error("No X, Y, Z, W gonosomes are present in the exon list.\n \
        Please check that the exon file initially a BED file matches the gonosomes processed here.")
        sys.exit(1) 
    return(gonoIndexDict, genderInfoList)

#############################
# clusterFromLinks
# Groups the samples according to their distance obtained with the hierarchical clustering method Average.
# Several conditions for forming clusters:
# -exceeding the minimum distance threshold ('minDist')
# -below the allowed distance ('maxDist')
# -number of samples greater than minSampsNbInCluster
# Decremental correlation method, if a single sample is grouped with a control cluster it is possible to 
# form a new cluster.
# Returns a negative validity status for clusters without control and without sufficient numbers.

# Args:
#  - FPMarray: is a float numpy array, dim = NbExons x NbSOIs 
#  - SOIs: is the str sample of interest name list 
#  - minDist: is a float variable (1-|r|), it's the minimal distance tolerated to start building clusters 
#  - maxDist: is a float variable, it's the maximal distance to concedered 
#  - minSampsInNbClust: is an int variable, defining the minimal sample number to validate a cluster
#  - figure: is a boolean: True or false to generate a figure
#  - outputFile: is a full path (+ file name) for saving a dendogram
# 
# Returns a 2D numpy array, dim= NbSOIs*3 columns: 
# 1) SOIs [str], samples names
# 2) clusterID [int], 
# 3) controlledBy [str], 
# 4) validitySamps [int], boolean 0: dubious sample and 1: valid sample

def clustersBuilds(FPMarray, SOIs, minDist, maxDist, minSampsNbInCluster, figure, outputFile):
    ###################################
    # part 1: Calcul distance between samples and apply hierachical clustering
    ###################################
    # Euclidean distance (classical method) not used
    # Absolute correlation distance is unlikely to be a sensible distance when 
    # clustering samples. ( 1-|r| where r is the Pearson correlation)
    correlation = np.round(np.corrcoef(FPMarray,rowvar=False),2)
    dissimilarity = 1 - abs(correlation) 

    # average linkage calcul
    # XXXX WHY?
    # squareform transform squared distance matrix in a triangular matrix
    # optimal_ordering: linkage matrix will be reordered so that the distance between
    # successive leaves is minimal.
    linksMatrix = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'average',optimal_ordering=True)

    ###################################
    # part 2: linksMatrix analysis, cluster formation and identification of controls/targets clusters
    ###################################
    ############
    # To Fill
    # clusters: a 1D numpy array, clusterID associated to SOIsIndex
    clusters=np.zeros(FPMarray.shape[1], dtype=np.int)
    # trgt2Ctrls: target and controls clusters correspondance,
    # key = target clusterID, value = list of controls clusterID 
    trgt2Ctrls={}  
    # links2Clusters: cluster formed thanks to the linksMatrix parsing associated with SOIs.
    # key = current clusterID, value = list of SOIs indexes 
    links2Clusters={}

    ############
    # To increment
    # identifies clusters
    # e.g first formed samples cluster in linksMatrix(NbSOIs-1) named 
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
            if (len(SOIsIndexInParents)<minSampsNbInCluster):
                clusters[SOIsIndexInParents] = clusterID

            # sufficient samples number
            else:
                ###############
                # Identification of the different cases 
                # Knowing that we are dealing with two parent clusters 
                
                # Case 1: both parent clusters have sufficient numbers to be controls
                # creation of a new target cluster with the current clusterID (trgt2Ctrls key)
                # the controls are the parents clusterID and their previous control clusterID (trgt2Ctrls list value)
                if ((nbSOIsInParents[0]>=minSampsNbInCluster) and (nbSOIsInParents[1]>=minSampsNbInCluster)):
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
    clusters, ctrls, validityStatus = STDZAndCheckResPrivate(clusters, trgt2Ctrls, minSampsNbInCluster)

    ###################################
    # part 4: Optionnal plot of a dendogram based on clustering results
    ###################################
    if figure:
        DendogramsPrivate(clusters, ctrls, linksMatrix, minDist, outputFile)
    
    ##########
    return(np.column_stack((SOIs, clusters, ctrls, validityStatus)))

###############################################################################
# gonosomeProcessing: 
# can distinguish gonosomes during clustering
# identifies organism gender from exons
# applies a no-priori approach (Kmeans) to separate samples into different group (only 2),
# based on normalised fragment counts of gonosomes 
# assigns gender to each predicted group
# performs two independent clustering
# Args:
#   -countsNorm: a float 2D numpy array of normalised counts, dim=NbExons*NbSOIs 
#   -SOIs: a str list of sample identifier
#   -gonoIndexDict: a dictionnary for correspondance between gonosomesID(key:str) and
#          exons indexes list (value:int)
#   -genderInfoList: a string list of lists, dim=NbGender*2columns [genderID[str], targetGonosomesID[str]]
#   -minSamples: an int variable for restrict the minimum number of samples for cluster formation 
#   -minLinks: a float variable to have a maximum distance selection threshold
#   -outFolder: a str variable indicating the path to the folder containing the results
#   -figure: a boolean variable to indicate that a graphical output is requested by the user

# Returns a list of lists, dim=nbSOIs*5columns
# [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int], genderPreds[str]]

def gonosomeProcessing(countsNorm, SOIs, gonoIndexDict, genderInfoList, minSamples, minLinks, outFolder, figure):
    # keeps counts of exons overlapping gonosomes
    gonoIndex = np.unique([item for sublist in list(gonoIndexDict.values()) for item in sublist]) 
    gonosomesFPM = np.take(countsNorm,gonoIndex,axis=0)
    
    ######################
    # Kmeans with k=2 (always)
    # transposition of gonosomesFPM to consider the samples
    kmeans = sklearn.cluster.KMeans(n_clusters=len(genderInfoList), random_state=0).fit(gonosomesFPM.T)
    #####################
    # coverage ratio calcul for the different Kmeans groups and on the different gonosomes 
    # can then associate group Kmeans with a gender
    # get a str list where indices are important because identical to groupKmeansID, contains genderID
    gender2KmeansGp=genderAttributionPrivate(kmeans, countsNorm,gonoIndexDict, genderInfoList)

    ####################
    # Independent clustering for the two Kmeans groups
    # returns for each group a list of lists, dim=NbSOIsInKmeansGp*4columns 
    # [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int]]
    sampsIndexG1 = np.where(kmeans.labels_==0)[0]
    gonosomesFPMG1 = gonosomesFPM[:,sampsIndexG1]
    targetSOIsG1 = [SOIs[i] for i in sampsIndexG1]
    outputFile = os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_gonosomes_"+gender2KmeansGp[0]+".png")
    resGono1 = clustersBuilds(gonosomesFPMG1, targetSOIsG1, minSamples, minLinks, figure, outputFile)

    sampsIndexG2 = np.where(kmeans.labels_==1)[0]
    gonosomesFPMG2 = gonosomesFPM[:,sampsIndexG2]
    targetSOIsG2 = [SOIs[i] for i in sampsIndexG2]
    outputFile = os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_gonosomes_"+gender2KmeansGp[1]+".png")
    resGono2 = clustersBuilds(gonosomesFPMG2, targetSOIsG2, minSamples, minLinks, figure, outputFile)

    ####################
    # Merge Results:
    # creation of the output list, dim=NbSOIs*5columns
    # browse the result tables independently but regulate according to the SOIs indices.
    # the clusterID for the second group is changed to not conflict with the clusterID of the first group,
    # add maxClusterIDGp1 to the identifier
    GonoClusters=[[]*5]*len(SOIs)
    maxClust=[]
    for row in range(len(resGono1)):
        sampID = resGono1[row][0]
        sampIndex = SOIs.index(sampID)
        rowToPrint = resGono1[row]
        rowToPrint.append(gender2KmeansGp[0])
        GonoClusters[sampIndex ] = rowToPrint
        cluster = resGono1[row][1]
        if (cluster in maxClust):
            continue
        else:
            maxClust.append(cluster)

    for row in range(len(resGono2)):
        sampID = resGono2[row][0]
        sampIndex = SOIs.index(sampID)
        rowToPrint = resGono2[row]
        rowToPrint[1] = max(maxClust)+rowToPrint[1]
        if (rowToPrint[2]!=""):
            for i in range(len(rowToPrint[2])):
                rowToPrint[2][i] = max(maxClust)+rowToPrint[2][i]
        rowToPrint.append(gender2KmeansGp[1])
        GonoClusters[sampIndex ] = rowToPrint
    return (GonoClusters)

###############################################################################
# printClustersFile:
# print the different types of outputs expected
# Args:
#   - 'resClustering' is a list of list, dim=NbSOIs*4columns 
#       [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int]
#       can be derived from clustering on all chromosomes or on the autosomes
#   - 'outFolder' is a str variable, it's the results folder path  
#   - 'resClusteringGonosomes' is a list of list, dim=NbSOIs*5columns
#       [sampleID[str], clusterID[int], controlledBy[int list], validitySamps[int], genderPreds[str]]
#       this argument is optional
#
# Print this data to stdout as a 'clustersFile'
def printClustersFile(resClustering, outFolder, resClusteringGonosomes=False):
    if resClusteringGonosomes:
        clusterFile=open(os.path.join(outFolder,"ResClustering_AutosomesAndGonosomes_"+str(len(resClustering))+"samples.tsv"),'w')
        sys.stdout = clusterFile
        toPrint = "samplesID\tclusterID_A\tcontrolledBy_A\tvaliditySamps_A\tgenderPreds\tclusterID_G\tcontrolledBy_G\tvaliditySamps_G"
        print(toPrint)
        for i in range(len(resClustering)):
            # SOIsID + clusterInfo for autosomes and gonosomes
            toPrint = resClustering[i][0]+"\t"+str(resClustering[i][1])+"\t"+",".join(map(str,resClustering[i][2]))+\
                "\t"+str(resClustering[i][3])+"\t"+resClusteringGonosomes[i][4]+"\t"+str(resClusteringGonosomes[i][1])+\
                    "\t"+",".join(map(str,resClusteringGonosomes[i][2]))+"\t"+str(resClusteringGonosomes[i][3])
            print(toPrint)
        sys.stdout = sys.__stdout__
        clusterFile.close()
    else:
        clusterFile=open(os.path.join(outFolder,"ResClustering_"+str(len(resClustering))+"samples.tsv"),'w')
        sys.stdout = clusterFile
        toPrint = "samplesID\tclusterID\tcontrolledBy\tvaliditySamps"
        print(toPrint)
        for i in range(len(resClustering)):
            # SOIsID + clusterInfo 
            toPrint = resClustering[i][0]+"\t"+str(resClustering[i][1])+"\t"+",".join(map(str,resClustering[i][2]))+"\t"+str(resClustering[i][3])
            print(toPrint)
        sys.stdout = sys.__stdout__
        clusterFile.close()

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###############################################################################
# genderAttributionPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Gender and group predicted by Kmeans matching 
# calcul of normalized count ratios per gonosomes and kmeans group
# ratio = median (normalized count sums list for a specific gonosome and for all samples in a kmean group)
# Args:
#  -kmeans: an int list of groupID predicted by Kmeans ordered on SOIsIndex 
#  -countsNorm: a float 2D numpy array of normalized count, dim=NbExons*NbSOIs
#  -gonoIndexDict: a dictionary of correspondence between gonosomeID[key:str] and exonsIndex[value:int list] 
#  -genderInfoList: a list of list, dim=NbGender*2columns ([genderID, specificChr])
# Returns a list of genderID where the indices match the groupID formed by the Kmeans.
# e.g ["F","M"], KmeansGp 0 = Female and KmeansGp 1 = Male

def genderAttributionPrivate(kmeans, countsNorm, gonoIndexDict, genderInfoList):
    # initiate float variable, 
    # save the first count ratio for a given gonosome
    previousCount=None

    # first browse on gonosome names (limited to 2)
    for gonoID in gonoIndexDict.keys():
        # int list of the current gonosome exons indexes 
        gonoExonIndexL=gonoIndexDict[gonoID]
        
        # second browse on the kmean groups (only 2)
        for kmeanGroup in np.unique(kmeans.labels_):
            # int list corresponding to the sample indexes in the current Kmean group
            SOIsIndexKGpL,=list(np.where(kmeans.labels_ == kmeanGroup))
            
            #####################
            # selection of specifics normalized count data
            gonoTmpArray=countsNorm[gonoExonIndexL,] # gonosome exons
            gonoTmpArray=gonoTmpArray[:,SOIsIndexKGpL] # Kmean group samples        
            
            #####################
            # ratio calcul (axis=0, sum all row/exons for each sample) 
            countRatio = np.median(np.sum(gonoTmpArray,axis=0))

            # Keep gender names in variables
            g1 = genderInfoList[0][0] #e.g human g1="M"
            g2 = genderInfoList[1][0] #e.g human g2="F"
            
            #####################
            # filling two lists corresponding to the gender assignment condition
            # condition1L and condition2L same construction
            # 1D string list , dim=2 genderID (indexes corresponds to Kmeans groupID) 
            if (previousCount!=None):    
                # row order in genderInfoList is important
                # 0 : gender with specific gonosome not present in other gender
                # 1 : gender with 2 copies of same gonosome
                if gonoID == genderInfoList[0][1]: # e.g human => M:chrY
                    #####################
                    # condition assignment gender number 1:
                    # e.g human case group of M, the chrY ratio is expected 
                    # to be higher than the group of F (>10*)
                    countsx10=10*countRatio
                    condition1L=[""]*2
                    if previousCount>countsx10:
                        condition1L[kmeanGroup-1]=g1 #e.g human Kmeans gp0 = M
                        condition1L[kmeanGroup]=g2 #e.g human Kmeans gp1 = F
                    else:
                        condition1L[kmeanGroup-1]=g2 #e.g human Kmeans gp0 = F
                        condition1L[kmeanGroup]=g1 #e.g human Kmeans gp1 = M
                else: # e.g human => F:chrX
                    #####################
                    # condition assignment gender number 2:
                    # e.g human case group of F, the chrX ratio should be in 
                    # the range of 1.5*ratiochrXM to 3*ratiochrXM
                    countsx1half=3*countRatio/2
                    condition2L=[""]*2
                    if previousCount>countsx1half and previousCount<2*countsx1half:
                        condition2L[kmeanGroup-1]=g2
                        condition2L[kmeanGroup]=g1
                    else:
                        condition2L[kmeanGroup-1]=g1
                        condition2L[kmeanGroup]=g2
            
                # restart for the next gonosome
                previousCount=None
            else:
                # It's the first ratio calculated for the current gonosome => saved for comparison 
                # with the next ratio calcul
                previousCount=countRatio
            
    # predictions test for both conditions
    # the two lists do not agree => raise an error and quit the process.
    if condition1L!=condition1L:
        logger.error("The conditions of gender allocation are not in agreement.\n \
            condition n°1, one gender is characterised by a specific gonosome: %s \n \
                condition n°2 that the other gender is characterised by 2 same gonosome copies: %s ", condition1L, condition2L)
        sys.exit(1)
    return(condition1L)

################################
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
        # we get index lists
        else:
            SOIsIndexInParents = SOIsIndexInParents+links2Clusters[parentID]
            nbSOIsInParents.append(len(links2Clusters[parentID])) 
    return(SOIsIndexInParents, nbSOIsInParents)

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