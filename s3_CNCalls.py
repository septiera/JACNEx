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
import clusterSamps.clustering
import CNCalls.CNCallsFile
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
    newClustsFile = ""
    outFile = ""
    # optionnal args with default values
    prevCNCallsFile = ""
    prevClustFile = ""
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
If a pre-existing copy number calls file (with --cncalls) produced by this program associated with
a previous clustering file are provided (with --prevclusts), extraction of the observation probabilities
for the samples in homogeneous clusters between the two versions, otherwise the copy number calls is performed.
In addition, all graphical support (pie chart of exon filtering per cluster) are
printed in pdf files created in plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --out [str] : file where results will be saved, must not pre-exist, will be gzipped if it ends
            with '.gz', can have a path component but the subdir must exist
    --prevcncalls [str] optional: pre-existing copy number calls file produced by this program,
            possibly gzipped, the observation probabilities of copy number types are copied
            for samples contained in immutable clusters between old and new versions of the clustering files.
    --prevclusts [str] optional: pre-existing clustering file produced by s2_clusterSamps.py for the same
            timestamp as the pre-existing copy number call file.
    --priors list[float]: prior probability for each copy number type in the order [CN0, CN1,CN2,CN3+].
            Must be passed as a comma separated string, default : """ + str(priors) + """
    --plotDir[str]: subdir (created if needed) where result plots files will be produced, default :  """ + str(plotDir) + """
    -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "out=", "prevcncalls=",
                                                           "prevclusts=", "priors=", "plotDir="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
        elif (opt in ("--clusts")):
            newClustsFile = value
        elif opt in ("--out"):
            outFile = value
        elif (opt in ("--prevcncalls")):
            prevCNCallsFile = value
        elif (opt in ("--prevclusts")):
            prevClustFile = value
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

    if newClustsFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(newClustsFile)):
        raise Exception("clustsFile " + newClustsFile + " doesn't exist.")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other args
    if (prevCNCallsFile != "" and prevClustFile == "") or (prevCNCallsFile == "" and prevClustFile != ""):
        raise Exception("you should not use --cncalls and --prevclusts alone but together. Try " + scriptName + " --help")

    if (prevCNCallsFile != "") and (not os.path.isfile(prevCNCallsFile)):
        raise Exception("CNCallsFile " + prevCNCallsFile + " doesn't exist")

    if (prevClustFile != "") and (not os.path.isfile(prevClustFile)):
        raise Exception("previous clustering File " + prevClustFile + " doesn't exist")

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
    return(countsFile, newClustsFile, outFile, prevCNCallsFile, prevClustFile, priors, plotDir)


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
    (countsFile, clustsFile, outFile, prevCNCallsFile, prevClustFile, priors, plotDir) = parseArgs(argv)

    # should density plots compare several different KDE bandwidth algorithms and values?
    # hard-coded here rather than set via parseArgs because this should only be set
    # to True for dev & testing
    testSmoothingBWs = False

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ######################################################
    # parse counts
    try:
        (exons, samples, countsArray) = countFrags.countsFile.parseCountsFile(countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed for %s : %s", countsFile, repr(e))
        raise Exception("parseCountsFile failed")

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # parse clusts
    try:
        (clusts2Samps, clusts2Ctrls, sex2Clust) = clusterSamps.clustering.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ######################################################
    # normalize counts (FPM)
    try:
        countsFPM = countFrags.countFragments.normalizeCounts(countsArray)
    except Exception as e:
        logger.error("normalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("normalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done normalizing counts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ######################################################
    # allocate CNcallsArray, and populate it with pre-calculated observed probabilities
    # if CNCallsFile and prevClustFile are provided.
    # also returns a boolean np.array to identify the samples to be reanalysed if the clusters change
    try:
        (CNcallsArray, callsFilled) = CNCalls.CNCallsFile.extractObservedProbsFromPrev(exons, samples, clusts2Samps, prevCNCallsFile, prevClustFile)
    except Exception as e:
        raise Exception("extractObservedProbsFromPrev failed - " + str(e))

    thisTime = time.time()
    logger.debug("Done parsing previous CNCallsFile and prevClustFile, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # total number of samples that still need to be processed
    nbOfSamplesToProcess = len(samples)
    for samplesIndex in range(len(samples)):
        if callsFilled[samplesIndex] and samplesIndex not in clusts2Ctrls["Samps_QCFailed"]:
            nbOfSamplesToProcess -= 1

    if nbOfSamplesToProcess == 0:
        logger.info("all provided samples are in previous callsFile and clusters are the same, not producing a new one")
    else:
        ####################################################
        # CN Calls
        try:
            futureRes = CNCalls.copyNumbersCalls.CNCalls(countsFPM, CNcallsArray, samples, callsFilled,
                                                         exons, sex2Clust, clusts2Samps, clusts2Ctrls, priors,
                                                         testSmoothingBWs, plotDir)
        except Exception:
            raise Exception("CNCalls failed")

        thisTime = time.time()
        logger.debug("Done Copy Number Calls, in %.2f s", thisTime - startTime)
        startTime = thisTime

        #####################################################
        # Print exon defs + calls to outFile
        CNCalls.CNCallsFile.printCNCallsFile(futureRes, exons, samples, outFile)

        thisTime = time.time()
        logger.debug("Done printing calls for all (non-failed) samples, in %.2fs", thisTime - startTime)
        logger.info("ALL DONE")


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
