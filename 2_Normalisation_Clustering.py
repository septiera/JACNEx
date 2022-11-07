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
import gzip
import time
import logging

# different scipy submodules are used for the application of hierachical clustering 
import scipy.cluster.hierarchy 
import scipy.spatial.distance  

# sklearn submodule is used to make clusters by Kmeans
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
logger.info("starting to work")
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
#FPMNormalisation:
#Fragment Per Million normalisation for comparison between samples
#NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
#Input:
#   -countsArray is a numpy array[exonIndex][sampleIndex] of counts [int]
#   -countsNorm is a numpy array[exonIndex][sampleIndex] of 0 [float]
#Output:
#   -countsNorm is a numpy array[exonIndex][sampleIndex] of FPM normalised counts [float]
@numba.njit
def FPMNormalisation(countsArray,countsNorm):
    for sampleCol in range(countsArray.shape[1]):
        SampleCountsSum=np.sum(countsArray[:,sampleCol])
        SampleCountNorm=(countsArray[:,sampleCol]*1e6)/SampleCountsSum #1e6 is equivalent to 1x10^6
        countsNorm[:,sampleCol]=SampleCountNorm
    return(countsNorm)

##############################################
#ReferenceBuilding:
# Identify sample clusters with the best correlation coefficient.
# Several conditions must be respected:
#   -the correlation between all samples in a cluster must be greater than 75%.
#   (1-75%) => must be less than 0.25
#   Some tools (such as ExomeDepth) set this threshold at 97%, below which 
#   they consider that the group cannot correctly predict CNVs (small cluster construction).
#   -a minimum number of samples composing the cluster fixed.
#   A baseline counting model with few samples is not optimal for CNVs calling.
#Inputs:
#   -FPMArray is a numpy array[exonIndex Of Gonosomes or Autosomes][sampleIndex] 
#   of normalized count[float]
#   -SOIs is a list of strings: the sample names of interest, ordered according to the input file
#   -minSampleInCluster [int] is the sample number needed to validate a cluster. 
#   This parameter can be specified by the user.
#Output:
#   - a list of list [sampleIndex] [4 columns]:
#       1-sampleName
#       2-cluster number
#       3-correlation threshold used to form the group
#       4-Boolean indicating if the group is optimal, that it meets the selection criteria.
#       0: valid sample , 1: dubious sample
def ReferenceBuilding(FPMArray,SOIs,minSampleInCluster):
    startTime=time.time()
    hashCluster=dict()
    #####################################################
    # Correlation:
    ##################
    #corrcoef return Pearson product-moment correlation coefficients.
    #rowvar=False parameter allows you to correlate the columns=sample and not the rows[exons].
    correlation=np.round(np.corrcoef(FPMArray,rowvar=False),2)

    ######################################################
    # Distance: 
    ##################
    # Euclidean distance (classical method) not used
    # Absolute correlation distance is unlikely to be a sensible distance when 
    # clustering samples. (1 - r where r is the Pearson correlation)
    dissimilarity = 1 - abs(correlation) 
    # Complete linkage, which is the more popular method, takes the maximum distance.
    # f=max(d(x,y)) ; x is a sample in one cluster and y is a sample in the other.
    sampleLinks = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'complete')

    ######################################################
    # Clustering :
    ##################
    #Correlation thresholds for group calculation
    #warning np.array in float type because the input matrix = 1-corr
    minThreshold=0.26
    maxThreshold=0.01
    thresholdRange=np.arange(maxThreshold,minThreshold,0.01)

    for corrThreshold in np.arange(maxThreshold,minThreshold,0.01):
        #round some floats that can be interpreted with a lot of decimal 
        # (e.g 0.060000000000000005 =0.06)
        #TODO find a best solution (inconclusive tests: np.arrange(dtype), np.linspace).
        corrThreshold=round(corrThreshold,2)     
        # fcluster : Form flat clusters from the hierarchical clustering defined by 
        # the given linkage matrix.
        # Return An 1D array[sampleIndex][i] is the flat cluster number to which 
        # original observation i belongs.
        groupsLabels=scipy.cluster.hierarchy.fcluster(sampleLinks, corrThreshold, criterion='distance')

        #creation of two tables with descriptive elements of the whole cluster for 
        #a single correlation threshold:
        #   -uniqueGroup which is a 1D array with the unique identifiers of the groups
        #   -countGroup which is a 1D array with sample counts for each group
        #The indices of the two tables correspond.
        uniqueCluster, countSampleCluster =np.unique(groupsLabels, return_counts=True)

        #Construction of reference groups.
        #If a trained group does not meet the conditions at the end of the loop, 
        # a warning message will be returned.  
        for clusterIndex in range(0,len(uniqueCluster)):
            #list of sample indices associated with this group
            groupLabelsIndex,=list(np.where(groupsLabels == uniqueCluster[clusterIndex]))
            # Construction of a new hash (hashCluster) from the clusters obtained with the 
            # highest correlation threshold. 
            # hashCluster[sampleIndex]="corrThreshold_group"
            if corrThreshold==maxThreshold:
                if countSampleCluster[clusterIndex]>=minSampleInCluster:
                    hashCluster=dict.fromkeys(groupLabelsIndex,str(corrThreshold)+"_"+str(uniqueCluster[clusterIndex])+"_0")
            #Complete the hashCluster
            else:
                keyslist=list(hashCluster.keys())
                if countSampleCluster[clusterIndex]>=minSampleInCluster:
                    #Test if all samples in current cluster are already in a cluster
                    #If the case, nothing is done.
                    #Otherwise all values for the samples are replaced in hashCluster.
                    intersect=set(groupLabelsIndex)&set(keyslist)
                    setdiff=set(groupLabelsIndex)-set(keyslist)
                    if len(intersect)<len(groupLabelsIndex):
                        for skipKey in groupLabelsIndex:
                            hashCluster[skipKey]=str(corrThreshold)+"_"+str(uniqueCluster[clusterIndex])+"_0"
                        
                #End of loop and not all patients are not in a cluster
                #creation of a non-optimal cluster with an alert
                elif countSampleCluster[clusterIndex]<minSampleInCluster and corrThreshold==np.round(thresholdRange[-1],2):
                    logger.warning("Creation of a sample cluster with insufficient numbers %s with low correlation %s",str(countSampleCluster[clusterIndex]),str(corrThreshold))
                    for skipKey in groupLabelsIndex:
                        hashCluster[skipKey]=str(corrThreshold)+"_"+str(uniqueCluster[clusterIndex])+"_1" 

    ######################################################
    # Output formatting:
    ##################
    # makes sure to get the sample names associated with a cluster number (between 0 and len(nbcluster)),
    # the associated correlation threshold and their reliability (1 reliable, 0 unreliable)
    hashClean=dict()
    groupCounter=0
    resList=[]
    for key in sorted(hashCluster):
        sample=SOIs[key]
        splitValue=hashCluster[key].split("_")
        if hashCluster[key] in hashClean.keys():
            resList.append([sample,hashClean[hashCluster[key]],splitValue[0],splitValue[2]])
        else:
            groupCounter+=1
            hashClean[hashCluster[key]]=groupCounter
            resList.append([sample,groupCounter,splitValue[0],splitValue[2]])
    thisTime=time.time()
    logger.debug("Done ReferenceBuilding in %.2f s", thisTime-startTime)
    return (resList)
    

##############################################
#parseMetadata:
#Inputs:
#   -metadataFH is a filehandle open for reading, content is a sample description file (2 columns) 
#        1-"sampleID" [str]
#        2-"Sex" [str] : gender information. (e.g human "M"=Male "F":Female)
#Outputs:
#   -hashSamp2Sex[SOIsSampleIndex]="Sex"
def parseMetadata(metadataFH,SOIs):
    startTime=time.time()
    ######################
    # parse header
    headers= metadataFH.readline().rstrip().split(",")
    hashSamp2Sex=dict()
    for line in metadataFH:
        splitLine=line.rstrip().split(",")
        sample=splitLine[headers.index("sampleID")]
        sex=splitLine[headers.index("Sex")]
        if sample in SOIs:
            hashSamp2Sex[SOIs.index(sample)]=sex
    thisTime=time.time()
    logger.debug("Done parseMetadata in %.2f s", thisTime-startTime)
    return(hashSamp2Sex)

################################################################################################
######################################## Main ##################################################
################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])

    ##########################################
    # parse user-provided arguments
    # mandatory args
    countsFile=""
    metadataFile=""
    bedFile=""
    ##########################################
    # optionnal arguments
    # default values fixed
    sexChromList=["chrX","chrY"]
    padding=10
    minSampleNBAutosomes=15
    minSampleNBGonosomes=12

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) 
and forms the reference groups for the call. 
Results are printed to stdout folder : 
- a TSV file format: first 4 columns hold the exon definitions, subsequent columns hold the normalised counts.
- a TSV file format: describe the distribution of samples in the reference groups (5 columns);
first column sample of interest (SOIs), second reference group number for autosomes, third the group to be compared 
for calling, fourth and fifth identical but for gonosomes.
ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns hold the fragment counts.
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column 
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --padding [int] : number of bps used to pad the exon coordinates, default : """+str(padding)+"""
   --metadata [str]: TSV file, contains 2 columns: "sampleID", "Sex". 
   --sexChrom [str]: a list of gonosome name [str]. (default ["chrX", "chrY"])
   --nbSampAuto [int]: an integer indicating the minimum sample number to create a reference cluster for autosomes.
                       (default =15)
   --nbSampGono [int]:same as previous variable but for gonosomes. (default =12)
   --out[str]: pre-existing folder to save the output files"""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","bed=","padding=","metadata=","sexChrom=","nbSampAuto=","nbSampGono=","out="])
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

        #### TO MODIFIED
        elif opt in ("--metadata"):
            metadataFile=value
            if not os.path.isfile(metadataFile):
                sys.exit("ERROR : metadataFile "+metadataFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--sexChrom"):
            sexChromList=value.split(",")# list building
            # sexChromList is checked later
        elif opt in ("--nbSampAuto"):
            try:
                minSampleNBAutosomes=np.int(value)
            except Exception as e:
                logger.error("Conversion of 'nbSampAuto' value to int failed : %s", e)
                sys.exit(1)
        elif opt in ("--nbSampGono"):
            try:
                minSampleNBGonosomes=np.int(value)
            except Exception as e:
                logger.error("Conversion of 'nbSampGono' value to int failed : %s", e)
                sys.exit(1)
        elif opt in ("--out"):
            outFolder=value
            if not os.path.isdir(outFolder):
                sys.exit("ERROR : outFolder "+outFolder+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    #Preparation:
    ##################
    # Preparation:
    # parse exons from BED
    try:
        exons=processBed(bedFile, padding)
    except Exception:
        logger.error("processBed failed")
        sys.exit(1)
        
    thisTime = time.time()
    logger.debug("Done pre-processing BED, in %.2f s", thisTime-startTime)
    startTime = thisTime

    # parse fragment counts from TSV
    try:
        if countsFile.endswith(".gz"):
            countsFH = gzip.open(countsFile, "rt")
        else:
            countsFH = open(countsFile,"r")
    except Exception as e:
        logger.error("Opening provided countsFile %s: %s", countsFile, e)
        raise Exception('cannot open countsFile')
     ######################
    # parse header from (old) countsFile
    sampleNames = countsFH.readline().rstrip().split("\t")
    # ignore exon definition headers "CHR", "START", "END", "EXON_ID"
    del sampleNames[0:4]

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

    ##################
    ###### Metadata parsing
    metadataName=os.path.basename(metadataFile)
    try:
        if metadataName.endswith(".gz"):
            metadataFH = gzip.open(metadataFile, "rt")
        else:
            metadataFH = open(metadataFile, "r")
    except Exception as e:
        logger.error("Opening provided CSV metadata file %s: %s", metadataFile, e)
        sys.exit(1)

    hashSamp2Sex=parseMetadata(metadataFH,sampleNames)

    #####################################################
    #Normalisation:
    ##################
    #create an empty array to filled with the normalized counts
    FPM=np.zeros((len(exons),len(sampleNames)),dtype=np.float32, order='F')
    
    #FPM calcul
    FPM=FPMNormalisation(countsArray,FPM)

    ##########
    ## write file in stdout normalisation TSV
    
    startTime = time.time()
    normalisationFile=open(outFolder+"/FPMCount_"+str(len(sampleNames))+"samples_"+time.strftime("%Y%m%d")+".tsv",'w')
    sys.stdout = normalisationFile
    toPrint = "CHR\tSTART\tEND\tEXON_ID\t"+"\t".join(sampleNames)
    print(toPrint)
    for index in range(len(exons)):
        toPrint=[]
        toPrint=[*exons[index],*np.round(FPM[index],2)]
        toPrint="\t".join(map(str,toPrint))
        print(toPrint)
    sys.stdout = sys.__stdout__
    normalisationFile.close()
    thisTime = time.time()
    logger.info("Writing normalised data in tsv file: in %.2f s", thisTime-startTime)
    
    #Dicotomisation of data associated with autosomes and gonosomes as correlations may
    # be influenced by gender. 
    gonosomeIndex=[0]
    autosomesFPM=np.delete(FPM,gonosomeIndex,axis=0)
    gonosomesFPM=np.take(FPM,gonosomeIndex,axis=0)

    #####################################################
    # Reference group selection:
    ##################
    logger.info("####### Autosomes process")
    autosomesCluster=ReferenceBuilding(autosomesFPM,sampleNames,minSampleNBAutosomes)

    #Gonosomes are a special case 
    # TODO how to adapt the script to other species?
    # It is necessary to have the gender information for each sample
    # But without appriori a Kmeans can be made to split the data on chromosome number
    # Works well on humans but needs to be tested on other species
    logger.info("####### Gonosomes process")
    kmeans =sklearn.cluster.KMeans(n_clusters=len(sexChromList), random_state=0).fit(gonosomesFPM.T)#transposition to consider the samples

    #check that all samples are clustered according to their gender
    #not the case returns a warning (in stderr) and stores the sampleIndex in a dubious sample list.
    #creation of two gender-specific lists with sampleIndices.
    Male=[]
    Female=[]
    dubiousSample=[]
    for index,clusterValue in enumerate(kmeans.labels_):
        sex=hashSamp2Sex[index]
        if sex=="M" and clusterValue==0:
            Male.append(index)
        elif sex=="F" and clusterValue==1:
            Female.append(index)
        else:
            logger.warning("%s does not cluster with its gender, dubious sample !!!",sampleNames[index])
            dubiousSample.append(index)
            if clusterValue==0:
                Male.append(index)
            else:
                Female.append(index)

    gonoMCluster=ReferenceBuilding(gonosomesFPM[:,Male],np.asarray(sampleNames)[Male].tolist(),minSampleNBGonosomes)
    gonoFCluster=ReferenceBuilding(gonosomesFPM[:,Female],np.asarray(sampleNames)[Female].tolist(),minSampleNBGonosomes)

    #####################################################
    # Print exon defs + counts to stdout
    startTime = time.time()
    clusterFile=open(outFolder+"/ReferenceCluster_"+str(len(sampleNames))+"samples_"+time.strftime("%Y%m%d")+".tsv",'w')
    sys.stdout = clusterFile
    toPrint = "sampleID\tAutosomesGP\tAutosomesCorr\tAutosomesValidSample\tGonosomesGP\tGonosomesCorr\tGonosomesValidSample"
    print(toPrint)
    maxGroupGonoM=max(l[1] for l in gonoMCluster)
    for sampleIndex in range(len(sampleNames)):
        toPrintAuto=[]
        toPrintAuto=autosomesCluster[sampleIndex]
        if sampleIndex in Male:
            idxInRes=Male.index(sampleIndex)
            listGono=gonoMCluster[idxInRes]
        else:
            idxInRes=Female.index(sampleIndex)
            listGono=gonoFCluster[idxInRes]
            listGono[1]=maxGroupGonoM+listGono[1]
        if sampleIndex in dubiousSample:
            listGono[3]="1"
        toPrint=[*toPrintAuto,*listGono[1:4]]
        toPrint="\t".join(map(str,toPrint))
        print(toPrint)
    sys.stdout = sys.__stdout__
    clusterFile.close()
    thisTime = time.time()
    logger.info("Writing reference cluster in tsv file: in %.2f s", thisTime-startTime)

if __name__ =='__main__':
    main()