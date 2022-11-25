###############################################################################################
######################################## MAGE-CNV step 3: Copy numbers calls ##################
###############################################################################################
# 
###############################################################################################
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import erf
import numba
import random
import time

####### MAGE-CNV modules
import mageCNV.countsFile
import mageCNV.normalisation
import mageCNV.genderDiscrimination

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of 
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    clustsFile = ""
    # optionnal args with default values
    


    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts and a TSV of reference group membership, 
normalize the counts and then call the copy numbers for each sample. 
The results are printed in an output folder where each sample will have its call
results in a TSV file. 
dim=NbExons*[(4 columns for exon information)+(4 columns for likelihood status for the copy number types 0, 1, 2 , 3+)   

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns 
                   hold the fragment counts. File obtained from 1_countFrags.py.
   --clusts [str]: TSV file, reference clusters (gender discrimination 8 columns, otherwise 4 columns)
                    dim: NbSamples*["sampleID","clusterID","controlledBy","validitySamps"]
                    added "genderPreds" column if gonosome analysis.
"""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","clusts="])
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : "+e.msg+". Try "+scriptName+" --help\n")
        raise Exception()

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            raise Exception()       
        elif (opt in ("--counts")):
            countsFile = value
            if (not os.path.isfile(countsFile)):
                sys.stderr.write("ERROR : countsFile "+countsFile+" doesn't exist.\n")
                raise Exception()    
        elif (opt in ("--clusts")):
            clustsFile = value
            if (not os.path.isfile(clustsFile)):
                sys.stderr.write("ERROR : clustsFile "+clustsFile+" doesn't exist.\n")
                raise Exception()    
        else:
            sys.stderr.write("ERROR : unhandled option "+opt+".\n")
            raise Exception()


    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        sys.exit("ERROR : You must use --counts.\n"+usage)
    if clustsFile == "":
        sys.exit("ERROR : You must use --clusts.\n"+usage)

    # AOK, return everything that's needed
    return(countsFile, clustsFile)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#####################################
# parseClustsFile
# extraction of clustering information
# Several cases are possible: a sample is part of a clustering on all the chromosomes or 
# it is part of several clusters, one associated with the autosomes and another associated with the gonosomes.
# Arg:
#  - clustsFile: a clusterFile produced by 2_clusterSamps.py 
# Returns a tupple (clusts2Samps,clusts2Ctrls) , each variable is created here:
#  - clusts2Samps: a str dictionary, key: clusterID , value: samples list 
#  - clusts2Ctrls: a str dictionary, key: clusterID, value: controlsID list
#  - nogender: a boolean to identify if a discrimination of gender is made
def parseClustsFile(clustsFile):
    try:
        clustsFH = open(clustsFile,"r")
    except Exception as e:
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception('cannot open clustsFile')

    # dictionary of clusterID and samples list 
    clusts2Samps={}

    # dictionary of clusterID and controlsID list
    clusts2Ctrls={}
    
    # variable to identify if a discrimination of gender is made
    nogender=False

    # grab headers
    headers= clustsFH.readline().rstrip().split("\t")
    
    # condition for identifying whether gender discrimination
    if (headers == ["samplesID","clusterID","controlledBy","validitySamps"]):
        for line in clustsFH:
            splitLine=line.rstrip().split("\t",maxsplit=3)
            # populate clust2Samps 
            # extract only valid samples
            if splitLine[3]==1:
                # init list value for the clusterID key
                if splitLine[0] not in clusts2Samps:
                    clusts2Samps[splitLine[0]]=[]
                clusts2Samps[splitLine[0]].append(splitLine[0])
                
                # populate clusts2Ctrls
                if splitLine[2]!="":
                    if splitLine[1] not in clusts2Ctrls:
                        clusts2Ctrls[splitLine[1]]=splitLine[2].split(",")  
            else:
                logger.warning("%s sample is dubious, it's not considered for the calling.")       
    
    elif (headers == ["samplesID","clusterID_A","controlledBy_A","validitySamps_A","genderPreds","clusterID_G","controlledBy_G","validitySamps_G"]):
        nogender=True
        for line in clustsFH:
            splitLine=line.rstrip().split("\t",maxsplit=7)
            # populate clust2Samps 
            # extract only valid samples for autosomes
            # tricky to eliminate samples on their validity associated with gonosomes, 
            # especially if one gender has few samples.
            if splitLine[3]=='1':
                #################
                # Autosomes
                # fill clusterID with fine description (A for autosomes,M for Male, F for Female) 
                newClustID=splitLine[1]+"A"
                if newClustID not in clusts2Samps:
                    clusts2Samps[newClustID]=[]
                clusts2Samps[newClustID].append(splitLine[0])
                
                if splitLine[2]!="":
                    if splitLine[1] not in clusts2Ctrls:
                        ctrlList = splitLine[2].split(",") 
                        ctrlList = list(map(( lambda x:x+'A'),ctrlList))
                        clusts2Ctrls[newClustID]=ctrlList  
                #################
                # Gonosomes
                newClustID=splitLine[1]+splitLine[4]
                if newClustID not in clusts2Samps:
                    clusts2Samps[newClustID]=[]
                clusts2Samps[newClustID].append(splitLine[0])
                
                if splitLine[6]!="":
                    if splitLine[1] not in clusts2Ctrls:
                        ctrlList = splitLine[6].split(",") 
                        ctrlList = list(map(( lambda x: x + splitLine[4]),ctrlList))
                        clusts2Ctrls[newClustID]=ctrlList

            else:
                logger.warning("%s sample is dubious for autosomes, it's not considered for the calling.",splitLine[0])     
    else:
        logger.error("Opening provided clustsFile %s: %s", clustsFile, headers)
        raise Exception('cannot open clustsFile')   
        
    return(clusts2Samps,clusts2Ctrls,nogender)

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    scriptName = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want scriptName rather than 'root'
    logger = logging.getLogger(scriptName)

    ################################################
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (countsFile, clustsFile) = parseArgs(argv) 

    ################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    #####################################################
    # Parse counts data:
    ################## 
    # parse counts from TSV to obtain :
    # - exons: a list of exons same as returned by processBed, ie each
    #    exon is a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
    #    copied from the first 4 columns of countsFile, in the same order
    # - SOIs: the list of sampleIDs (ie strings) copied from countsFile's header
    # - countsArray: an int numpy array, dim = NbExons x NbSOIs
    try:
        (exons, SOIs, countsArray) = mageCNV.countsFile.parseCountsFile(countsFile)
    except Exception:
        logger.error("parseCountsFile failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Parse clusts data:
    ################## 
    # parse clusters from TSV to obtain:
    # - clusts2Samps: a str dictionary, key: clusterID , value: samples list 
    # - clusts2Ctrls: a str dictionary, key: clusterID, value: controlsID list
    try:
        (clusts2Samps, clusts2Ctrls, nogender) = parseClustsFile(clustsFile)
    except Exception:
        logger.error("parseClustsFile failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Normalisation:
    ##################  
    # allocate a float numpy array countsNorm and populate it with normalised counts of countsArray
    # same dimension for arrays in input/output: NbExons*NbSOIs
    # Fragment Per Million normalisation:
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    try :
        countsNorm = mageCNV.normalisation.FPMNormalisation(countsArray)
    except Exception:
        logger.error("FPMNormalisation failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done fragments counts normalisation, in %.2f s", thisTime-startTime)
    startTime = thisTime

    if nogender:
        #######################################################
        # Extract Gender Informations
        ##################
        # parse exons to extract information related to the organisms studied and their gender
        # gonoIndex: is a dictionary where key=GonosomeID(e.g 'chrX')[str], 
        # value=list of gonosome exon index [int].
        # genderInfos: is a str list of lists, contains informations for the gender
        # identification, ie ["gender identifier","particular chromosome"].
        try: 
            (gonoIndex, genderInfo) = mageCNV.genderDiscrimination.getGenderInfos(exons)
        except Exception: 
            logger.error("getGenderInfos failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime-startTime)
        startTime = thisTime

        ##############
        # cutting normalized count data according to autosomal or gonosomal exons
        # create flat gonosome index list
        gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist]) 
        autosomesFPM = np.delete(countsNorm,gonoIndexFlat,axis=0)
        gonosomesFPM = np.take(countsNorm,gonoIndexFlat,axis=0)











    else:


####################################################################################
######################################## Main ######################################
####################################################################################

if __name__ =='__main__':
    try:
        main(sys.argv)
    except Exception:
        # whoever raised the exception should have explained it on stderr, here we just die
        exit(1)