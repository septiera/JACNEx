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
# parse the bed to obtain a list of lists (dim=NbExon x [CHR,START,END,EXONID])
# the exons are sorted according to their genomic position and padded
from countFrags.bed import processBed 

# parse a pre-existing counts file
from countFrags.oldCountsFile import parseCountsFile 

################################################################################################
################################ Functions #####################################################
################################################################################################
##############################################
# FPMNormalisation:
# Fragment Per Million normalisation for comparison between samples
# NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
# Inputs:
#   -countsArray : numpy array of counts [int] (Dim=[exonIndex]x[sampleIndex])
#   -countsNorm : numpy array of 0 [float] (Dim=[exonIndex]x[sampleIndex])
# Output:
#   -countsNorm : numpy array of FPM normalised counts [float] (Dim=[exonIndex]x[sampleIndex])
@numba.njit
def FPMNormalisation(countsArray,countsNorm):
    for sampleCol in range(countsArray.shape[1]):
        SampleCountsSum=np.sum(countsArray[:,sampleCol])
        SampleCountNorm=(countsArray[:,sampleCol]*1e6)/SampleCountsSum #1e6 is equivalent to 1x10^6
        countsNorm[:,sampleCol]=SampleCountNorm
    return(countsNorm)    

##############################################
# clusterBuilding :
# Group samples with similar coverage profiles (FPM standardised counts).
# Use absolute pearson correlation and hierarchical clustering.
# Args:
#   -FPMArray is a float numpy array, dim = NbExons x NbSOIs 
#   -minSampleInCluster is an int variable, defining the minimal sample number to validate a cluster
#   -minLinks is a float variable, it's the minimal distance tolerated for build clusters 
#   (advice:run the script once with the graphical output to deduce this threshold as specific to the data)

#Return:2 dictionnairy and a list of ints
#   - SOI2ClusterDict: key: Sample Index [int], value: clusterID [int]
#   - controlClusterDict: key: control clusterID [int], value: target(s) clusterID ints list
#   - inValidSOIs : dubious sample of interest indexes list [int]

def clusterBuilding(FPMArray, minSampleInCluster, minLinks):
    #####################################################
    # Correlation:
    ##################
    #corrcoef return Pearson product-moment correlation coefficients.
    #rowvar=False parameter allows you to correlate the columns=sample and not the rows[exons].
    correlation=np.round(np.corrcoef(FPMArray,rowvar=False),2) #correlation value decimal < -10² = tricky to interpret
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
    clusterCounter=0
    
    # To fill
    # sample membership in a cluster
    # key: Sample Index [int], value: clusterID [int]
    SOI2ClusterDict=dict()
    # control cluster to be used for other clusters
    # key: control clusterID [int], value: target(s) clusterID int list
    controlClusterDict=dict()
    # enerating dubious data
    inValidSOIs=[]
    
    # Distances range definition 
    # add 0.01 to minLinks as np.arrange(start,stop,step) don't conciderate stop in range 
    distanceStep=0.05
    linksRange=np.arange(distanceStep,minLinks+0.01,distanceStep)
    
    for selectLinks in linksRange:
        # fcluster : Form flat clusters from the hierarchical clustering defined by 
        # the given linkage matrix.
        # Return An 1D array, dim= sampleIndex[groupNb]
        groupFormedList=scipy.cluster.hierarchy.fcluster(samplesLinks, selectLinks, criterion='distance')

        # a 1D array with the unique identifiers of the groups
        uniqueClusterID=np.unique(groupFormedList)
        
        ######################
        # Cluster construction
        # all clusters obtained for a distance value (selectLinks) are processed
        for clusterIndex in range(len(uniqueClusterID)):
            # list of sample indexes associated with this group
            SOIsIndexInCluster,=list(np.where(groupFormedList == uniqueClusterID[clusterIndex]))
            
            #####################
            # Group selection criterion, enough samples number in cluster
            if len(SOIsIndexInCluster)>=minSampleInCluster:
                # contains the control groups ID of the current group [int list]
                gpControlList=[]
                
                ######################
                # part filling the SOI2ClusterDict
                ######################
                # all samples in the current cluster are processed
                # GOALS: identification of samples to be added if new, 
                # keep clusterID for samples already seen
                
                # New samples indexes list
                SOIToAddList=set(SOIsIndexInCluster)-set(SOI2ClusterDict.keys())
                
                # New cluster if new samples are present
                # For clusters with the same composition from one distance threshold to the other,
                # no analysis is performed                                       
                if len(SOIToAddList)!=0:
                    clusterCounter+=1
                    for SOIIndex in SOIsIndexInCluster:
                        if SOIIndex in SOIToAddList:                           
                             SOI2ClusterDict[SOIIndex]=clusterCounter 
                        else:
                            SOIgp=SOI2ClusterDict[SOIIndex]
                            if SOIgp not in gpControlList:
                                gpControlList.append(SOIgp)
                            #clusterID already identified, next sample                             
                            else:
                                continue     
                # no new sample move to next cluster
                else:
                    continue
                                
                ######################
                # part filling the controlClusterDict
                ######################
                if len(gpControlList)!=0:
                    for gp in gpControlList:
                        if gp in controlClusterDict:
                            tmpList=controlClusterDict[gp]
                            tmpList.append(clusterCounter)
                            controlClusterDict[gp]=tmpList
                        else:
                            controlClusterDict[gp]=[clusterCounter]
            
            # not enough samples number in cluster              
            else: 
                #####################
                # Case where it's the last distance threshold and the cluster is not large enough
                # New cluster creation with dubious sample for calling step 
                if selectLinks==linksRange[-1]:
                    clusterCounter+=1
                    logger.warning("Creation of cluster n°%s with insufficient numbers %s\
                         with low correlation %s",clusterCounter,str(len(SOIsIndexInCluster)),str(selectLinks))
                    for SOIIndex in SOIsIndexInCluster:
                        SOI2ClusterDict[SOIIndex]=clusterCounter
                        inValidSOIs.append(SOIIndex)
                else:
                    continue
    return(SOI2ClusterDict,controlClusterDict,inValidSOIs)

################################################################################################
######################################## Main ##################################################
################################################################################################
def main():

    scriptName=os.path.basename(sys.argv[0])

    ##########################################
    # parse user-provided arguments
    # mandatory args
    countsFile=""
    bedFile=""
    ##########################################
    # optionnal arguments
    # default values fixed
    padding=10
    genotypes=[["male","X","Y"],["female","X","X"]]
    minSample=20
    minLinks=0.25

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and a TSV of exon fragment counts, normalizes the counts 
(Fragment Per Million) and forms the reference groups for the call. 
Results are printed to stdout folder : 
- a TSV file format: first 4 columns hold the exon definitions, subsequent columns hold the normalised counts.
- a TSV file format: describe the distribution of samples in the reference groups (5 columns);
  first column sample of interest (SOIs), second reference group number for autosomes, third the group to be compared 
  for calling, fourth and fifth identical but for gonosomes.
ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns 
                   hold the fragment counts.
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column 
                   headerless tab-separated file, columns contain CHR START END EXON_ID)
   --padding [int] : number of bps used to pad the exon coordinates, default : """+str(padding)+""" 
   --genotypes [str]: comma-separated list of list fo sexual genotypes, default : """+str(genotypes)+"""
   --genotypes-from [str]: text file listing sexual genotypes, one per line. (format : not fixed columns number
                  1:genotypeName 2:Gonosome n°1 3:Gonosome n°2)
   --minSample [int]: an integer indicating the minimum sample number to create a reference cluster for autosomes,
                  default : """+str(minSample)+"""
   --minLinks [float]: a float indicating the minimal distance to considered for the hierarchical clustering,
                  default : """+str(minLinks)+"""                                          
   --out[str]: pre-existing folder to save the output files"""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","bed=","padding=","genotypes=","genotypes-from=","minSample=","minLinks=","out="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--counts"):
            countsFile=value
            if not os.path.isfile(countsFile):
                sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--bed"):
            bedFile=value
            if not os.path.isfile(bedFile):
                sys.exit("ERROR : bedFile "+bedFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--padding"):
            try:
                padding=np.int(value)
            except Exception as e:
                logger.error("Conversion of padding value to int failed : %s", e)
                sys.exit(1)
        elif opt in ("--genotypes"):
            genotypes =value
            # genotypes is checked later
        elif opt in ("--genotypes-from"):
            genotypesFrom=value
            if not os.path.isfile(genotypesFrom):
                sys.exit("ERROR : genotypes-from file "+genotypesFrom+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--minSample"):
            try:
                minSample=np.int(value)
            except Exception as e:
                logger.error("Conversion of 'minSample' value to int failed : %s", e)
                sys.exit(1)
        elif opt in ("--minLinks"):
            try:
                minLinks=np.float(value)
            except Exception as e:
                logger.error("Conversion of 'minLinks' value to int failed : %s", e)
                sys.exit(1)
        elif opt in ("--out"):
            outFolder=value
            if not os.path.isdir(outFolder):
                sys.exit("ERROR : outFolder "+outFolder+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    # Check that the mandatory parameters are present
    if countsFile=="":
        sys.exit("ERROR : You must use --counts.\n"+usage)
    if bedFile=="":
        sys.exit("ERROR : You must use --bed.\n"+usage)
    
    ######################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    #####################################################
    # Preparation:
    ##################
    # parse exons from BED
    try:
        exons=processBed(bedFile, padding)
    except Exception:
        logger.error("processBed failed")
        sys.exit(1)
        
    thisTime = time.time()
    logger.debug("Done pre-processing BED, in %.2f s", thisTime-startTime)
    startTime = thisTime

    ####################
    # parse fragment counts from TSV
    # Header extraction of the count file to find the names and the samples number
    # needed for parsing
    with open(countsFile) as f:
        sampleNames= f.readline().rstrip().split("\t")
        del sampleNames[0:4]
    logger.debug("Samples number to be treated : %s", len(sampleNames))

    # countsArray[exonIndex][sampleIndex] will store the corresponding count.
    # order=F should improve performance, since we fill the array one column at a time.
    # dtype=np.uint32 should also be faster and totally sufficient to store the counts
    # defined as a global variable for simplified filling during parallelization. 
    countsArray = np.zeros((len(exons),len(sampleNames)),dtype=np.uint32, order='F')
    # countsFilled: same size and order as sampleNames, value will be set 
    # to True iff counts were filled from countsFile
    countsFilled = np.array([False]*len(sampleNames))

    # fill countsArray with pre-calculated counts if countsFile was provided
    if (countsFile!=""):
        try:
            parseCountsFile(countsFile,exons,sampleNames,countsArray,countsFilled)
        except Exception as e:
            logger.error("parseCountsFile failed - %s", e)
            sys.exit(1)

        thisTime = time.time()
        logger.debug("Done parsing old countsFile, in %.2f s", thisTime-startTime)
        startTime = thisTime

    #####################################################
    # Check and clean up the genotypes lists
    # list containing the sex chromosomes, with any dupes removed
    gonosomes=[]

    if (genotypesFrom!=""):
        genotypes=[]
        genoList = open(genotypesFrom,"r")
        for line in genoList:
            lineInfo = line.rstrip()
            lineInfo= lineInfo.split(",")
            genotypes.append(lineInfo)
    
    # It should be taken into account the different non-human organisms 
    genoTmp=np.array(genotypes)
    for row in range(len(genoTmp)): #genotypes is unlimited
        for col in range(len(genoTmp[row])-1): #several chromosomes can induce the genotype (not only 2)
            if not genoTmp[row][col+1].startswith("chr"):
                genoTmp[row][col+1]="chr"+genoTmp[row][col+1]
            if genoTmp[row][col+1] not in gonosomes:
                gonosomes.append(genoTmp[row][col+1])

    # Check that gonosomes exist in exons and extract gonosomes exons index
    gonoIndex=[]
    controlDict={}
    for exonId in range(len(exons)):
        if exons[exonId][0] in gonosomes:
            gonoIndex.append(exonId)
            if exons[exonId][0] in controlDict:
                controlDict[exons[exonId][0]] += 1
            else:
                controlDict[exons[exonId][0]] =1
        else:
            continue

    for gonosome in gonosomes:
        if gonosome not in controlDict:
            logger.error("Gonosome %s not defined in bedFile.Please check.\n \
                        Tip: for all non-numerary gonosomes processBed has annotated them maxCHR+1.(except for X and Y) ",gonosome)
        else:
            continue
    #####################################################
    # Normalisation:
    ##################
    #create an empty array to filled with the normalized counts
    FPM=np.zeros((len(exons),len(sampleNames)),dtype=np.float32, order='F')
    
    #FPM calcul
    FPM=FPMNormalisation(countsArray,FPM)
    thisTime = time.time()
    logger.debug("Done FPM normalisation, in %.2f s", thisTime-startTime)
    startTime = thisTime

    ##################
    ## write file in stdout normalisation TSV
    # alert no solution found to speed up saving
    # numba => no successful conversion
    # round() goes faster (78,74s, here 104.17s)

    # output file definition
    normalisationFile=open(outFolder+"/FPMCounts_"+str(len(sampleNames))+"samples_"+time.strftime("%Y%m%d")+".tsv",'w')
    #replace basic sys.stdout by the file to be filled
    sys.stdout = normalisationFile 

    toPrint = "CHR\tSTART\tEND\tEXON_ID\t"+"\t".join(sampleNames)
    print(toPrint)
    for index in range(len(exons)):
        toPrint=[]
        toPrint=[*exons[index],*["%.2f" % x for x in FPM[index]]]# method selected to remain consistent with previous scripts
        toPrint="\t".join(map(str,toPrint))
        print(toPrint)
    normalisationFile.close()
    thisTime = time.time()
    logger.info("Writing normalised data in tsv file: in %.2f s", thisTime-startTime)
    #sys.stdout reassigning to sys.__stdout__
    sys.stdout = sys.__stdout__
    
    # Dicotomisation of data associated with autosomes and gonosomes as correlations may
    # be influenced by gender. 
    autosomesFPM=np.delete(FPM,gonoIndex,axis=0)
    gonosomesFPM=np.take(FPM,gonoIndex,axis=0)
    
    ################
    # Get Autosomes Clusters
    SOI2ClusterAutosomes,controlClusterAutosomes,inValidSOIsAutosomes=clusterBuilding(autosomesFPM,minSample, minLinks)

if __name__ =='__main__':
    main()