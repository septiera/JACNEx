###############################################################################################
######################################## MAGE-CNV step 3: Copy numbers calls ##################
###############################################################################################
# Given a TSV file containing the clustering results assign logOdds for each copy number
# (0,1,2,3+) per exon.
# Produces one TSV file per sample.
###############################################################################################
import sys
import getopt
import os
import time
import logging

####### MAGE-CNV modules
import mageCNV.countsFile
import mageCNV.normalisation
import mageCNV.genderDiscrimination
import mageCNV.slidingWindow
import mageCNV.copyNumbersCalls


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
Given a TSV file containing the clustering results assign logOdds for each copy number
(0,1,2,3+) per exon.
Produces one TSV file per sample.
dim=NbExons*(exon informations[CHR,START,END,EXONID])+(4 columns LogOdds for copy number types [0, 1, 2 , 3+])

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py.
   --clusts [str]: TSV file, reference clusters. File obtained from 2_clusterSamps.py.
                   2 expected format:
                        - no gender discrimination, dim=NbSOIs*5columns
                        - gender discrimination, dim=NbSOIs*9columns""" + "\n"

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts="])
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : " + e.msg + ". Try " + scriptName + " --help\n")
        raise Exception()

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            raise Exception()
        elif (opt in ("--counts")):
            countsFile = value
            if (not os.path.isfile(countsFile)):
                sys.stderr.write("ERROR : countsFile " + countsFile + " doesn't exist.\n")
                raise Exception()
        elif (opt in ("--clusts")):
            clustsFile = value
            if (not os.path.isfile(clustsFile)):
                sys.stderr.write("ERROR : clustsFile " + clustsFile + " doesn't exist.\n")
                raise Exception()
        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        sys.exit("ERROR : You must use --counts.\n" + usage)
    if clustsFile == "":
        sys.exit("ERROR : You must use --clusts.\n" + usage)

    # AOK, return everything that's needed
    return(countsFile, clustsFile)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):

    ################################################
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (countsFile, clustsFile) = parseArgs(argv)

    ################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    #####################################################
    # Parse counts data :
    ##################
    # parse counts from TSV to obtain :
    # - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
    # - SOIs (list[str]): sampleIDs copied from countsFile's header
    # - countsArray (np.ndarray[int]): fragment counts, dim = NbExons x NbSOIs
    try:
        (exons, SOIs, countsArray) = mageCNV.countsFile.parseCountsFile(countsFile)
    except Exception:
        logger.error("parseCountsFile failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Parse clusts data :
    ##################
    # parse clusters from TSV to obtain:
    # - clusts2Samps: a str dictionary, key: clusterID , value: samples list
    # - clusts2Ctrls: a str dictionary, key: clusterID, value: controlsID list
    # - nogender (boolean): identify if a discrimination of gender is made
    try:
        (clusts2Samps, clusts2Ctrls, nogender) = mageCNV.copyNumbersCalls.parseClustsFile(clustsFile, SOIs)
    except Exception:
        logger.error("parseClustsFile failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Normalisation:
    ##################  
    # Fragment Per Million normalisation:
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    # this normalisation allows the samples to be compared
    # - countsNorm (np.ndarray[float]): normalised counts of countsArray same dimension
    # for arrays in input/output: NbExons*NbSOIs
    try :
        counts_norm = mageCNV.normalisation.FPMNormalisation(countsArray)
    except Exception:
        logger.error("FPMNormalisation failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done fragments counts normalisation, in %.2f s", thisTime - startTime)
    startTime = thisTime
    
    #######################################################
    # Extract Gender Informations
    ##################
    # parse exons to extract information related to the organisms studied and their gender
    # - gonoIndex (dict(str: list(int))): is a dictionary where key=GonosomeID(e.g 'chrX'),
    # value=list of gonosome exon index.
    # - genderInfo (list of list[str]):contains informations for the gender
    # identification, ie ["gender identifier","specific chromosome"].
    try: 
        (gonosome_index, gender_info) = mageCNV.genderDiscrimination.getGenderInfos(exons)
    except Exception: 
        logger.error("getGenderInfos failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done get gender informations in %.2f s", thisTime - startTime)
    startTime = thisTime
    
    
####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    try:
        main(sys.argv)
    except Exception:
        # whoever raised the exception should have explained it on stderr, here we just die
        exit(1)
