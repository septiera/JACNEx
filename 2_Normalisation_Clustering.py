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
   --out[str]: pre-existing folder to save the output files"""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","bed=","padding=","out="])
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
    #Preparation:
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
    #Normalisation:
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
    
    

if __name__ =='__main__':
    main()