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

###############################################################################
# clusterBuilds :
# Group samples with similar coverage profiles (FPM standardised counts).
# Use absolute pearson correlation and hierarchical clustering.
# Args:
#   -FPMArray is a float numpy array, dim = NbExons x NbSOIs 
#   -SOIs is the str sample of interest name list 
#   -minSampleInCluster is an int variable, defining the minimal sample number to validate a cluster
#   -minLinks is a float variable, it's the minimal distance tolerated for build clusters 
#   (advice:run the script once with the graphical output to deduce this threshold as specific to the data)
#   -figure: is a boolean: True or false to generate a figure
#   -outputFile: is a full path (+ file name) for saving a dendogram

#Return :
# -resClustering: a list of list with different columns typing, it's the clustering results,
# the list indexes are ordered according to the SOIsIndex
#  dim= NbSOIs*4 columns: 
# 1) sampleName [str], 
# 2) clusterID [int], 
# 3) controlledBy [ints list], 
# 4) validitySamps [int], boolean 0: dubious sample and 1: valid sample,

# -[optionally] a png showing a dendogram

def clusterBuilds(FPMArray, SOIs, minSampleInCluster, minLinks, figure, outputFile):
    #####################################################
    # Correlation:
    ##################
    #corrcoef return Pearson product-moment correlation coefficients.
    #rowvar=False parameter allows you to correlate the columns=sample and not the rows[exons].
    correlation = np.round(np.corrcoef(FPMArray,rowvar=False),2) #correlation value decimal < -10² = tricky to interpret
    ######################################################
    # Distance: 
    ##################
    # Euclidean distance (classical method) not used
    # Absolute correlation distance is unlikely to be a sensible distance when 
    # clustering samples. (1 - r where r is the Pearson correlation)
    dissimilarity = 1 - abs(correlation) 
    # Complete linkage, which is the more popular method,
    # Distances between the most dissimilar members for each pair of clusters are calculated
    # and then clusters are merged based on the shortest distance
    # f=max(d(x,y)) ; x is a sample in one cluster and y is a sample in the other.
    # DEV WARNING : probably to be changed to a more sophisticated method like Ward
    # scipy.squareform is necessary to convert a vector-form distance vector to a square-form 
    # distance matrix readable by scipy.linkage
    samplesLinks = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'complete')
    
    ######################################################
    # Clustering :
    ##################
    # Iteration on decreasing distance values every 0.05.
    # Cluster formation if its size is greater than minSamplesInCluster.
    # Formation of the most correlated clusters first, they will be used as a control
    # for the clusters obtained at the next lower distance value.
    # For the last iteration (lowest distance), if the number of clusters is not 
    # sufficient, a new cluster is created, but return a warning message for the clusterID and 
    # an invalidity status for the samples concerned.
        
    # Accumulators:
    # Attribute cluster identifier [int]
    clusterCounter = 0
    
    # To fill
    # a list of list with different types of arguments [str,int,intsList,int]
    resClustering=[0]*len(SOIs)
    # a dictionary with the correpondence between control cluster (key: int) 
    # and cluster to be controlled (value: int list)
    controlsDict={}

    # Distances range definition 
    # add 0.01 to minLinks as np.arrange(start,stop,step) don't conciderate stop in range 
    distanceStep = 0.05
    linksRange = np.arange(distanceStep,minLinks+0.01,distanceStep)
 
    for selectLinks in linksRange:
        # fcluster : Form flat clusters from the hierarchical clustering defined by 
        # the given linkage matrix.
        # Return An 1D array, dim= sampleIndex[clusterNb]
        clusterFormedList = scipy.cluster.hierarchy.fcluster(samplesLinks, selectLinks, criterion='distance')

        # a list of the unique identifiers of the clusters [int]
        uniqueClusterID = np.unique(clusterFormedList)
        
        ######################
        # Cluster construction
        # all clusters obtained for a distance value (selectLinks) are processed
        for clusterIndex in range(len(uniqueClusterID)):
            # list of sample indexes associated with this cluster
            SOIIndexInCluster, = list(np.where(clusterFormedList == uniqueClusterID[clusterIndex]))
            
            #####################
            # Cluster selection criterion, enough samples number in cluster
            if (len(SOIIndexInCluster)>=minSampleInCluster):
                # New samplesIndexes to fill in resClustering list
                SOIIndexToAddList = [index for index,value in enumerate(resClustering) 
                                   if (index in SOIIndexInCluster and value==0)]
                
                # Create a new cluster if new samples are presents                                    
                if (len(SOIIndexToAddList)!=0):
                    clusterCounter += 1
                    # selection of samples indexes already present in older clusters => controls clusters 
                    IndexToGetClustInfo = set(SOIIndexInCluster)-set(SOIIndexToAddList)
                    # list containing all unique clusterID as control for this new cluster [int]
                    listCtrlClust = set([value[1] for index,value in enumerate(resClustering) 
                                       if (index in IndexToGetClustInfo)])
                    # Filling resClustering for new samples
                    for SOIIndex in SOIIndexToAddList:
                        resClustering[SOIIndex]=[SOIs[SOIIndex],clusterCounter,list(listCtrlClust),1]
                    # Filling controlsDict if new cluster is controlled by previous cluster(s)
                    for ctrl in listCtrlClust:
                        if ctrl in controlsDict:
                            tmpList=controlsDict[ctrl]
                            tmpList.append(clusterCounter)
                            controlsDict[ctrl]=tmpList
                        else:
                            controlsDict[ctrl]=[clusterCounter]
                    
                # probably a previous formed cluster, same sample composition, no analysis is performed
                else:
                    continue
            # not enough samples number in cluster              
            else: 
                #####################
                # Case where it's the last distance threshold and the samples have never been seen
                # New cluster creation with dubious samples for calling step 
                if (selectLinks==linksRange[-1]):
                    clusterCounter += 1
                    logger.warning("Creation of cluster n°%s with insufficient numbers %s with low correlation %s",
                    clusterCounter,str(len(SOIIndexInCluster)),str(selectLinks))
                    for SOIIndex in SOIIndexInCluster:
                        resClustering[SOIIndex] = [SOIs[SOIIndex],clusterCounter,list(),0]
                else:
                    continue  
    # case we wish to obtain the graphical representation
    if figure:
        try:
            clusterDendogramsPrivate(resClustering, controlsDict,clusterCounter, samplesLinks, outputFile)
        except Exception as e: 
            logger.error("clusterDendograms failed - %s", e)
            sys.exit(1)

    return(resClustering)

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
    resGono1 = clusterBuilds(gonosomesFPMG1, targetSOIsG1, minSamples, minLinks, figure, outputFile)

    sampsIndexG2 = np.where(kmeans.labels_==1)[0]
    gonosomesFPMG2 = gonosomesFPM[:,sampsIndexG2]
    targetSOIsG2 = [SOIs[i] for i in sampsIndexG2]
    outputFile = os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_gonosomes_"+gender2KmeansGp[1]+".png")
    resGono2 = clusterBuilds(gonosomesFPMG2, targetSOIsG2, minSamples, minLinks, figure, outputFile)

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

def genderAttributionPrivate(kmeans, countsNorm,gonoIndexDict, genderInfoList):
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

###############################################################################
# clusterDendogramsPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# visualisation of clustering results
# Args:
# -resClustering: a list of list with dim: NbSOIs*4 columns 
# ["sampleName"[str], "ClusterID"[int], "controlsClustID"[int], "ValiditySamps"[int]]
# -maxClust: a int variable for maximum clusterNb obtained after clustering
# -controlsDict: a dictionary key:controlsCluster[int], value:list of clusterID to Control[int]  
# -sampleLinks: a 2D numpy array of floats, correspond to the linkage matrix, dim= NbSOIs-1*4. 
# -outputFile : full path to save the png 
# Return a png to the user-defined output folder

def clusterDendogramsPrivate(resClustering, controlsDict,maxClust, sampleLinks, outputFile):
    #initialization of a list of strings to complete
    labelsGp=[]

    # Fill labelsgp at each new sample/row in resClustering
    # Cluster numbers link with the marking index in the label 
    # 2 marking type: "*"= target cluster ,"-"= control cluster present in controlsDict
    for sIndex in range(len(resClustering)):
        # an empty str list of length of the maximum cluster found
        tmpLabels=["   "]*maxClust
        row=resClustering[sIndex] 
        # select group-related position in tmplabels
        #-1 because the group numbering starts at 1 and not 0
        indexCluster=row[1]-1
        # add symbol to say that this sample is a target 
        tmpLabels[indexCluster]=" * "
        # case the sample is part of a control group
        # add symbol to the index of groups to be controlled in tmpLabels
        if row[1] in controlsDict:
            for gpToCTRL in controlsDict[row[1]]:
                # symbol add to tmplabels for the current group index.
                tmpLabels[gpToCTRL-1]=" - "
        labelsGp.append("".join(tmpLabels))

    # dendogram plot
    matplotlib.pyplot.figure(figsize=(5,15),facecolor="white")
    matplotlib.pyplot.title("Complete linkage hierarchical clustering")
    dn1 = scipy.cluster.hierarchy.dendrogram(sampleLinks,orientation='left',
                                             labels=labelsGp, 
                                             color_threshold=0.05)
    
    matplotlib.pyplot.ylabel("Samples of interest")
    matplotlib.pyplot.xlabel("Absolute Pearson distance (1-r)")
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="png", bbox_inches='tight')