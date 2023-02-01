###############################################################################################
######################################## MAGE-CNV step 3: Copy numbers calls ##################
###############################################################################################
# Given a TSV of exon fragment counts and a TSV with clustering information,
# calculation of emission probabilities (logOdds) for each copy number type (CN0, CN1, CN2, CN3+).
# The results are printed in the stdout in TSV format: the first 4 columns contain
# the definitions of the paddled and sorted exons and the following columns
# (4 per sample) contain the logOdds.
# In addition, all graphical support (pie chart of exon filtering per cluster) are
# printed in pdf files created in plotDir.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import time
import logging
import numpy as np

####### MAGE-CNV modules
import countFrags.countsFile
import countFrags.countFragments
import clusterSamps.clustering
import CNCalls.copyNumbersCalls


# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)

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
    priors = "0.001,0.01,0.979,0.01"
    plotDir = "./ResultPlots/"

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts and a TSV with clustering information,
calculation of emission probabilities (logOdds) for each copy number type (CN0, CN1, CN2, CN3+).
The results are printed in the stdout in TSV format: the first 4 columns contain
the definitions of the paddled and sorted exons and the following columns
(4 per sample) contain the logOdds.
In addition, all graphical support (pie chart of exon filtering per cluster) are
printed in pdf files created in plotDir.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py.
   --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
                   File obtained from 2_clusterSamps.py.
   --priors list[float]: prior probability for each copy number type in the order [CN0, CN1,CN2,CN3+].
                         Must be passed as a comma separated string parameter, default : """ + str(priors) + """
   --plotDir[str]: subdir (created if needed) where result plots files will be produced, default :  """ + str(plotDir) + """
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "priors=", "plotDir="])
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
        elif (opt in ("--clusts")):
            clustsFile = value
        elif (opt in ("--priors")):
            priors = value
        elif (opt in ("--plotDir")):
            plotDir = value
        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        sys.exit("ERROR : You must provide a counts file with --counts. Try " + scriptName + " --help.\n")
        raise Exception()
    elif (not os.path.isfile(countsFile)):
        sys.stderr.write("ERROR : countsFile " + countsFile + " doesn't exist.\n")
        raise Exception()

    if clustsFile == "":
        sys.exit("ERROR : You must provide a clustering results file use --clusts. Try " + scriptName + " --help.\n")
        raise Exception()
    elif (not os.path.isfile(clustsFile)):
        sys.stderr.write("ERROR : clustsFile " + clustsFile + " doesn't exist.\n")
        raise Exception()

    #####################################################
    # Check other args
    if priors == "":
        sys.stderr.write("ERROR : You must provide four correlation values separated by commas with --priors. Try " + scriptName + " --help.\n")
        raise Exception()
    else:
        try:
            priors = [float(x) for x in priors.split(",")]
            if (sum(priors) != 1):
                raise Exception()
        except Exception:
            sys.stderr.write("ERROR : priors must be four float numbers whose sum is 1, not '" + priors + "'.\n")
            raise Exception()

    # test plotDir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception:
            sys.stderr.write("ERROR : plotDir " + plotDir + " doesn't exist and can't be mkdir'd\n")
            raise Exception()

    # AOK, return everything that's needed
    return(countsFile, clustsFile, priors, plotDir)


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
    (countsFile, clustsFile, priors, plotDir) = parseArgs(argv)

    ################################################
    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
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
        (exons, SOIs, countsArray) = countFrags.countsFile.parseCountsFile(countsFile)
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
    # - clusts2Samps (dict[str, List[int]]): key: clusterID , value: samples index list based on SOIs list
    # - clusts2Ctrls (dict[str, List[str]]): key: clusterID, value: controlsID list
    # - SampsQCFailed list[str] : sample names that failed QC
    # - sex2Clust dict[str, list[str]]: key: "A" autosomes or "G" gonosome, value: clusterID list
    try:
        (clusts2Samps, clusts2Ctrls, SampsQCFailed, sex2Clust) = CNCalls.copyNumbersCalls.parseClustsFile(clustsFile, SOIs)
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
    try:
        countsNorm = clusterSamps.normalisation.FPMNormalisation(countsArray)
    except Exception:
        logger.error("FPMNormalisation failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done fragments counts normalisation, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################################################
    # CN Calls
    ####################
    # initiate results object
    # - Returns an all zeroes float array, adapted for
    # storing the logOdds for each type of copy number.
    # dim= NbExons x [NbSOIs x [CN0, CN1, CN2,CN3+]]
    logOddsArray = CNCalls.copyNumbersCalls.allocateLogOddsArray(SOIs, exons)

    try:
        logOddsArray = CNCalls.copyNumbersCalls.CNCalls(sex2Clust, exons, countsNorm, clusts2Samps, clusts2Ctrls, priors, SOIs, plotDir, logOddsArray)

    except Exception:
        logger.error("CNCalls failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done Copy Number Calls, in %.2f s", thisTime - startTime)
    startTime = thisTime


####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))

    try:
        main(sys.argv)
    except Exception:
        # whoever raised the exception should have explained it on stderr, here we just die
        exit(1)
