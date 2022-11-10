###############################################################################################
######################################## MAGE-CNV step 2: Normalisation & clustering ##########
###############################################################################################
#  Given a BED of exons and a TSV of exon fragment counts,
#  normalizes the counts (Fragment Per Million) and forms the reference groups for the call. 
#  Prints results in a folder defined by the user. 
#  See usage for more details.
###############################################################################################

import sys
import getopt
import os
import numpy as np
import numba
import time
import logging

# different scipy submodules are used for the application of hierachical clustering 
import scipy.cluster.hierarchy 
import scipy.spatial.distance  

# import sklearn submodule for  Kmeans calculation
import sklearn.cluster

# prevent numba DEBUG messages filling the logs when we are in DEBUG loglevel
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# !!!! definition of the logger here as the functions are not yet modularised (will change) 
# configure logging, sub-modules will inherit this config
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))

################################################################################################
################################ Modules #######################################################
################################################################################################
# parse a pre-existing counts file
from mageCNV.countsFile import parseCountsFile 

################################################################################################
################################ Functions #####################################################
################################################################################################
##############################################
# extractGonosomesInfos:
# Evaluation of arguments passed by user 
# Arg:
# - gender: a str list of list, each list corresponds to a gender, ie format : ["genderName","chrG1","chrG2",etc ]
# - exons: 
# Returns two dictionaries:
# - genderInfosDict: key: gender name [str], value: chromosomes list [str]
# - sexChrToBedIndexDict: key: sexChromName [str], value: list of indexes in bedfile [int]
def extractGonosomesInfos(gender,exons):
    # genderInfosDict building
    # Takes into account the different non-human organisms 
    # no fixed gender number
    # no fixed chromosomes number
    genderInfosDict = dict()
    for row in gender:
        tmpChrList=[] # will contain all chromosomes specified for the current gender
        for i in range(len(row)):
            # Key definition
            if (i == 0):
                genderName = row[i]
            # Value definition
            # checking the input syntax format
            # force to include "chr" before the chromosome ID
            # nothing happens if it already has
            else:
                if (not row[i].startswith("chr")):
                    addChr = "chr"+row[i]
                    tmpChrList.append(addChr)
                else:
                    tmpChrList.append(row[i])
            # filling dictionary with corrects infos
            genderInfosDict[genderName] = tmpChrList

    # sexChrToBedIndexDict building
    # is used to extract the exon indexes specific to each sex chromosome
    sexChrToBedIndexDict = dict()
    # list containing only unique chromosome identifiers for all gender
    SexChrFlatList = np.unique([item for sublist in list(genderInfosDict.values()) for item in sublist])
    for exonId in range(len(exons)):
        if (exons[exonId][0] in SexChrFlatList):
            if (exons[exonId][0] in sexChrToBedIndexDict):
                sexChrToBedIndexDict[exons[exonId][0]].append(exonId)
            else:
                sexChrToBedIndexDict[exons[exonId][0]] = [exonId]
        else:
            continue

    # Checks if the chromosomes are found in the bed file 
    # if not returns an error and break process
    if (len(set(SexChrFlatList)-set(sexChrToBedIndexDict))!=0):
        logger.error("Gonosome %s not defined in bedFile.Please check.\n \
            Tip: for all non-numerary gonosomes processBed has annotated them maxCHR+1.(except for X and Y) ",
            set(SexChrFlatList)-set(sexChrToBedIndexDict))
        sys.exit(1)   
    return(genderInfosDict,sexChrToBedIndexDict)

##############################################
# FPMNormalisation:
# Fragment Per Million normalisation => comparison between samples
# NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
# Inputs:
#   -countsArray : numpy array of counts [int] (Dim=[exonIndex]x[sampleIndex])
#   -countsNorm : numpy array of 0 [float] (Dim=[exonIndex]x[sampleIndex])
# Output:
#   -countsNorm : numpy array of FPM normalised counts [float] , Dim=exonIndex*sampleIndex
@numba.njit
def FPMNormalisation(countsArray):
    #create an empty array to filled with the normalized counts
    countsNorm = np.zeros_like(countsArray,dtype=np.float32)
    for sampleCol in range(countsArray.shape[1]):
        SampleCountsSum = np.sum(countsArray[:,sampleCol])
        SampleCountNorm = (countsArray[:,sampleCol]*1e6)/SampleCountsSum #1e6 is equivalent to 1x10^6
        countsNorm[:,sampleCol] = SampleCountNorm
    return(countsNorm)    

##############################################
# clusterBuilding :
# Group samples with similar coverage profiles (FPM standardised counts).
# Use absolute pearson correlation and hierarchical clustering.
# Args:
#   -FPMArray is a float numpy array, dim = NbExons x NbSOIs 
#   -SOIs is the str sample of interest name list 
#   -minSampleInCluster is an int variable, defining the minimal sample number to validate a cluster
#   -minLinks is a float variable, it's the minimal distance tolerated for build clusters 
#   (advice:run the script once with the graphical output to deduce this threshold as specific to the data)

#Return :
# -resClustering: a 2D numpy array with different columns typing, extract clustering results,
#  dim= NbSOIs*5. columns: 0) SOIs Index [int],1) SOIs Names [str], 2)clusterID [int], 
#  3)clusterIDToControl [str], 4) Sample validity for calling [int] 
# -samplesLinks: a 2D numpy array of floats, correspond to the linkage matrix, dim= NbSOIs-1*4

def clusterBuilding(FPMArray, SOIs, minSampleInCluster, minLinks):
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
    # 2D np array with different column typing, dim = NbSOIs*5columns
    resClustering = np.full((len(SOIs),5),(np.int,"",np.int,"",1))
   
    # Distances range definition 
    # add 0.01 to minLinks as np.arrange(start,stop,step) don't conciderate stop in range 
    distanceStep = 0.05
    linksRange = np.arange(distanceStep,minLinks+0.01,distanceStep)
 
    for selectLinks in linksRange:
        # fcluster : Form flat clusters from the hierarchical clustering defined by 
        # the given linkage matrix.
        # Return An 1D array, dim= sampleIndex[groupNb]
        groupFormedList = scipy.cluster.hierarchy.fcluster(samplesLinks, selectLinks, criterion='distance')

        # a 1D array with the unique identifiers of the groups
        uniqueClusterID = np.unique(groupFormedList)
        
        ######################
        # Cluster construction
        # all clusters obtained for a distance value (selectLinks) are processed
        for clusterIndex in range(len(uniqueClusterID)):
            # list of sample indexes associated with this group
            SOIIndexInCluster, = list(np.where(groupFormedList == uniqueClusterID[clusterIndex]))
            
            #####################
            # Group selection criterion, enough samples number in cluster
            if (len(SOIIndexInCluster)>=minSampleInCluster):
                # New samples to fill in numpy array, it's list of SOIs indexes
                SOIToAddList = set(SOIIndexInCluster)-set(resClustering[:,0])
                
                # New cluster if new samples are presents
                # For clusters with the same composition from one distance threshold to the other,
                # no analysis is performed                                       
                if (len(SOIToAddList)!=0):
                    clusterCounter += 1
                    for SOIIndex in SOIIndexInCluster:
                        if (SOIIndex in SOIToAddList):
                            # fill informations for new patient
                            resClustering[SOIIndex,0] = SOIIndex
                            resClustering[SOIIndex,1] = SOIs[SOIIndex]
                            resClustering[SOIIndex,2] = clusterCounter 
                        else:
                            # update informations for patient used as control
                            # need to pass countClusters in str for the final format 
                            if (resClustering[SOIIndex,3]!=""):
                                ControlGp = resClustering[SOIIndex,3].split(",")
                                if (str(clusterCounter) not in ControlGp):
                                    ControlGp.append(str(clusterCounter))
                                    resClustering[SOIIndex,3] = ",".join(ControlGp)
                                else:
                                    continue
                            else:
                                resClustering[SOIIndex,3] = str(clusterCounter)
            
            # not enough samples number in cluster              
            else: 
                #####################
                # Case where it's the last distance threshold and the samples have never been seen
                # New cluster creation with dubious samples for calling step 
                if (selectLinks==linksRange[-1]):
                    clusterCounter+=1
                    logger.warning("Creation of cluster n°%s with insufficient numbers %s with low correlation %s",
                    clusterCounter,str(len(SOIIndexInCluster)),str(selectLinks))
                    for SOIIndex in SOIIndexInCluster:
                        resClustering[SOIIndex,0] = SOIIndex
                        resClustering[SOIIndex,1] = SOIs[SOIIndex]
                        resClustering[SOIIndex,2] = clusterCounter 
                        resClustering[SOIIndex,4] = 0
                else:
                    continue  
    return(resClustering,samplesLinks)

################################################################################################
######################################## Main ##################################################
################################################################################################
def main():
    scriptName = os.path.basename(sys.argv[0])
    ##########################################
    # parse user-provided arguments
    # mandatory args
    countsFile = ""
    ##########################################
    # optionnal arguments
    # default values fixed
    minSamples = 20
    minLinks = 0.25
    nogender = False
    figure = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and forms the reference 
groups for the call. 
Separation of autosomes and gonosomes (chr accepted: X, Y, Z, W) for clustering, to avoid bias.
Results are printed to stdout folder:
- a TSV file format, describe the clustering results, dim = NbSOIs*8 columns:
    1) sample of interest (SOIs),
    2) groupID for autosomes, 
    3) groupID controlling the current group for autosomes,
    4) sample validity for autosomes to considered for calling steps, 0=dubious and 1=valid,
    5) gender predicted by Kmeans, M=Male and F=Female,
    6) groupID for gonosomes, 
    7) groupID of controls for gonosomes,
    8) sample validity for gonosomes.
- one or more png's illustrating the clustering performed on the data by dendograms. [optionnal]
    Legend : solid line = target groups , thin line = control groups
    The groups appear in decreasing order of distance.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns 
                   hold the fragment counts.
   --minSamples [int]: an integer indicating the minimum sample number to create a reference cluster for autosomes,
                  default : """+str(minSamples)+""".
   --minLinks [float]: a float indicating the minimal distance to considered for the hierarchical clustering,
                  default : """+str(minLinks)+""".   
   --nogender: no autosomes and gonosomes discrimination for clustering.
   --figure: make one or more dendograms that will be present in the output in png format.                                          
   --out[str]: pre-existing folder to save the output files"""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","minSamples=","minLinks=","nogender=","figure=","out="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
            if (not os.path.isfile(countsFile)):
                sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif (opt in ("--minSamples")):
            try:
                minSamples = np.int(value)
            except Exception as e:
                logger.error("Conversion of 'minSamples' value to int failed : %s", e)
                sys.exit(1)
        elif (opt in ("--minLinks")):
            try:
                minLinks = np.float(value)
            except Exception as e:
                logger.error("Conversion of 'minLinks' value to int failed : %s", e)
                sys.exit(1)
        elif (opt in ("--nogender")):
            nogender = True
        elif (opt in ("--figure")):
            figure = True
        elif (opt in ("--out")):
            outFolder = value
            if not os.path.isdir(outFolder):
                sys.exit("ERROR : outFolder "+outFolder+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    # Check that the mandatory parameters are present
    if countsFile == "":
        sys.exit("ERROR : You must use --counts.\n"+usage)
    
    ######################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    # parse counts from TSV to obtain :
    # - a list of exons same as returned by processBed, ie each
    #    exon is a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
    #    copied from the first 4 columns of countsFile, in the same order
    # - the list of sampleIDs (ie strings) copied from countsFile's header
    # - an int numpy array, dim = len(exons) x len(samples)
    try:
        exons, SOIs, countsArray = parseCountsFile(countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed - %s", e)
        sys.exit(1)

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime
    
    #####################################################
    # Check the gender arguments
    # case where a file is placed as an argument by the user
    # creation of a list of lists in the same format as "gender" 
    # i.e. [genderName, chr1, chr2, ...].
    if (genderFrom!=""):
        gender = []
        try:
            genderLists = open(genderFrom,"r")
        except Exception as e:
            logger.error("Opening genotypeFrom failed - %s", e)
            sys.exit(1)
        for genderline in genderLists :
            genderList = genderline.rstrip()
            genderList = genderList.split(",")
            gender.append(genderList)

    # Check and clean up the gender lists to obtain two dictionaries:
    # - genderInfosDict: key: gender name [str], value: chromosomes list [str]
    # - sexChrToBedIndexDict: key: sexChromName [str], value: list of indexes in bedfile [str]
    try :
        genderInfosDict,sexChrToBedIndexDict = extractGonosomesInfos(gender,exons)
    except Exception as e:
        logger.error("extractGonosomesInfos failed - %s", e)
        sys.exit(1)

    thisTime = time.time()
    logger.debug("Done gonosomes information extraction, in %.2f s", thisTime-startTime)
    startTime = thisTime
   
    #####################################################
    # Normalisation:
    ##################  
    # allocate countsNorm and populate it with normalised counts of countsArray
    # same dimension for arrays in input/output: exonIndex*sampleIndex
    # Fragment Per Million normalisation
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    try :
        countsNorm = FPMNormalisation(countsArray)
    except Exception as e:
        logger.error("FPMNormalisation failed - %s", e)
        sys.exit(1)
    thisTime = time.time()
    logger.debug("Done FPM normalisation, in %.2f s", thisTime-startTime)
    startTime = thisTime

    # Dicotomisation of data associated with autosomes and gonosomes 
    # Goals: the correlations calculated for clustering are not biased
    gonoIndex = np.unique([item for sublist in list(sexChrToBedIndexDict.values()) for item in sublist])
    autosomesFPM = np.delete(countsNorm,gonoIndex,axis=0)
    gonosomesFPM = np.take(countsNorm,gonoIndex,axis=0)
    
    #####################################################
    # Get Autosomes Clusters
    ##################
    # Application of hierarchical clustering on autosome count data to obtain :
    # - a 2D numpy array with different columns typing, extract clustering results,
    #       dim= NbSOIs*5. columns: 0) SOIs Index [int],1) SOIs Names [str], 2)clusterID [int], 
    #       3)clusterIDToControl [str], 4) Sample validity for calling [int] 
    # - a 2D numpy array of floats, correspond to the linkage matrix, dim= NbSOIs-1*4. 
    #       Required for the graphical part.
    logger.info("### Samples clustering on normalised counts of autosomes")
    try :
        resClusteringAutosomes, sampleLinksAutosomes = clusterBuilding(autosomesFPM, SOIs, minSamples, minLinks)
    except Exception as e:
        logger.error("clusterBuilding for autosomes failed - %s", e)
        sys.exit(1)
    thisTime = time.time()
    logger.info("Done samples clustering for autosomes : in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Get Gonosomes Clusters
    ##################
    # Different treatment
    # It is necessary to have the gender genotype information
    # But without appriori a Kmeans can be made to split the data on gender number
    logger.info("### Samples clustering on normalised counts of gonosomes")
    kmeans =sklearn.cluster.KMeans(n_clusters=len(genderInfosDict.keys()), random_state=0).fit(gonosomesFPM.T)#transposition to consider the samples


if __name__ =='__main__':
    main()