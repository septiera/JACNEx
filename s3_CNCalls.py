###############################################################################################
######################################## MAGE-CNV step 3: Copy numbers calls ##################
###############################################################################################
# Given a TSV of exon fragment counts and a TSV with clustering information,
# normalize the counts (in FPM = fragments per million), obtaining the observation 
# probabilities per copy number (CN), per exon and for each sample. 
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import time
import logging

####### MAGE-CNV modules
import countFrags.countsFile
import countFrags.countFragments
import CNCalls.copyNumbersCalls

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
# If anything is wrong, raise Exception("ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    clustsFile = ""
    # optionnal args with default values
    priors = "0.001,0.01,0.979,0.01"
    plotDir = "./ResultPlots/"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts and a TSV with clustering information,
normalize the counts (Fragments Per Million), deduces the copy numbers (CN) observation
probabilities, per exon and for each sample.
Results are printed to stdout in TSV format (possibly gzipped): first 4 columns hold the exon
definitions padded and sorted, subsequent columns (four per sample, in order CN0,CN2,CN2,CN3+)
hold the observation probabilities.
In addition, all graphical support (pie chart of exon filtering per cluster) are
printed in pdf files created in plotDir.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py.
   --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
                   [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
                   File obtained from 2_clusterSamps.py.
   --priors list[float]: prior probability for each copy number type in the order [CN0, CN1,CN2,CN3+].
                         Must be passed as a comma separated string, default : """ + str(priors) + """
   --plotDir[str]: subdir (created if needed) where result plots files will be produced, default :  """ + str(plotDir) + """
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "priors=", "plotDir="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
        elif (opt in ("--clusts")):
            clustsFile = value
        elif (opt in ("--priors")):
            priors = value
        elif (opt in ("--plotDir")):
            plotDir = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    if clustsFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(clustsFile)):
        raise Exception("clustsFile " + clustsFile + " doesn't exist.")

    #####################################################
    # Check other args
    try:
        priors = [float(x) for x in priors.split(",")]
        if (sum(priors) != 1) or (len(priors) != 4):
            raise Exception()
    except Exception:
        raise Exception("priors must be four float numbers whose sum is 1, not '" + priors + "'.")
    
    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, clustsFile, priors, plotDir)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    try:
        (countsFile, clustsFile, priors, plotDir) = parseArgs(argv)
    except Exception:
        # problem is described in Exception, just re-raise
        raise

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse counts
    try:
        (exons, SOIs, countsArray) = countFrags.countsFile.parseCountsFile(countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed for %s : %s", countsFile, repr(e))
        raise Exception("parseCountsFile failed")

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # parse clusts
    try:
        (clusts2Samps, clusts2Ctrls, SampsQCFailed, sex2Clust) = CNCalls.copyNumbersCalls.parseClustsFile(clustsFile, SOIs)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ###################
    # normalize counts (FPM)
    try:
        countsFPM = countFrags.countFragments.normalizeCounts(countsArray)
    except Exception as e:
        logger.error("normalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("normalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done normalizing counts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ####################################################
    # CN Calls
    ####################
    #
    #
    #
    try:
        emissionArray = CNCalls.copyNumbersCalls.CNCalls(sex2Clust, exons, countsFPM, clusts2Samps, clusts2Ctrls, priors, SOIs, plotDir)
    except Exception:
        raise Exception("CNCalls failed")

    thisTime = time.time()
    logger.debug("Done Copy Number Calls, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################################################
    # print results
    ####################
    #
    #
    #
    #
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
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + sys.argv[0] + " : " + str(e) + "\n")
        sys.exit(1)
