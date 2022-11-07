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
        elif opt in ("--out"):
            outFolder=value
            if not os.path.isdir(outFolder):
                sys.exit("ERROR : outFolder "+outFolder+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    ######################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()
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
    

if __name__ =='__main__':
    main()