###############################################################################################
######################################## MAGE-CNV step 3: Copy numbers calls ##################
###############################################################################################
# Given a TSV of exon fragment counts and a TSV with clustering information,
# obtaining the observation probabilities per copy number (CN), per exon and for each sample.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import math
import time
import logging
from concurrent.futures import ProcessPoolExecutor

####### MAGE-CNV modules
import countFrags.countsFile
import clusterSamps.getGonosomesExonsIndexes
import clusterSamps.clustFile
import clusterSamps.clustering
import CNCalls.CNCallsFile
import CNCalls.copyNumbersCalls
import CNCalls.userTSVFile

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
    clustFile = ""
    outFile = ""
    # optionnal args with default values
    samps2CheckFile = ""
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Given a TSV file containing exon fragment counts and a TSV file containing clustering information.
It deduces the observation probabilities of copy numbers (CN) for each exon and each sample.
Results are displayed in TSV format on stdout, with exon definitions in the first four columns
and probabilities in the subsequent columns (four columns per sample, in the order CN0, CN1, CN2, CN3+).
Graphical support, such as coverage plots and pie charts, is printed in PDF files.
These files are saved in analyzed cluster folders created in the output call TSV folder.
Additional coverage plots can be generated in debug mode.
If a TSV file containing sample IDs with targeted exons (--samps2check) is provided,
coverage profile plots will be generated.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
            with '.gz', can have a path component but the subdir must exist
    --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    --samps2check [str]: TSV file, contains 2 columns the sample name and exon identifier according to the bed
                         file supplied in step 1. File created by the user.
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "out=",
                                                           "jobs=", "samps2check="])
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
            clustFile = value
        elif opt in ("--out"):
            outFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif (opt in ("--samps2check")):
            samps2CheckFile = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    if clustFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(clustFile)):
        raise Exception("clustsFile " + clustFile + " doesn't exist.")

    #####################################################
    # Check other argsjobs = round(0.8 * len(os.sched_getaffinity(0)))
    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        if samps2CheckFile != "":
            raise Exception("samps2Check should not be provided when the logger level is not set to DEBUG.")

    # AOK, return everything that's needed
    return(countsFile, clustFile, outFile, jobs, samps2CheckFile)


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
    (countsFile, clustsFile, outFile, jobs, samps2CheckFile) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse counts, perform FPM normalization, distinguish between intergenic regions and exons
    try:
        (samples, exons, intergenics, exonsFPM, intergenicsFPM) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # parse clusts
    try:
        (clusters, ctrlsClusters, validity, specClusters) = clusterSamps.clustFile.parseClustsFile(clustsFile, samples)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################################################
    # parse user-defined tsv file of target samples and exons
    samps2Check = []
    if samps2CheckFile != "":
        try:
            samps2Check = CNCalls.userTSVFile.parseUserTSV(samps2CheckFile, samples, exons)
        except Exception as e:
            raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    ######################################################
    # init calling process
    # we have chosen to call copy numbers by exon, defining 4 copy number types:
    # CN0: total loss of copies, CN1: single copy, CN2: two copies (normal),
    # and CN3: three or more copies.
    CNTypes = ["CN0", "CN1", "CN2", "CN3"]

    CNCallsArray = CNCalls.CNCallsFile.allocateCNCallsArray(len(exons), len(samples), len(CNTypes))

    ####################################################
    # CN Calls
    ##############################
    # decide how the work will be parallelized
    # we are allowed to use jobs cores in total: we will process paraClusters clusters in
    # parallel
    # -> we target targetCoresPerCluster, this is increased if we
    #    have few clusters to process (and we use ceil() so we may slighty overconsume)
    paraClusters = min(math.ceil(jobs), len(clusters))
    logger.info("%i new clusters => will process %i in parallel", len(clusters), paraClusters)

    # identifying autosomes and gonosomes "exons" index
    # recall clusters are derived from autosomal or gonosomal analyses
    try:
        exonsBool = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)
    except Exception as e:
        raise Exception("getSexChrIndexes failed %s", e)

    #####################################################
    # Define nested callback for processing CNCalls() result (so CNCallsArray et al
    # are in its scope)

    # mergeCalls:
    # arg: a Future object returned by ProcessPoolExecutor.submit(CNCalls.copyNumbersCalls.CNCalls).
    # CNCalls() returns a 4-element tuple (clustID, colIndInCNCallsArray, sourceExons, clusterCalls).
    # If something went wrong, log and populate failedClusters;
    # otherwise fill column at index colIndInCNCallsArray and row at index sourceExons in CNCallsArray
    # with calls stored in clusterCalls
    def mergeCalls(futurecounts2callsRes):
        e = futurecounts2callsRes.exception()
        if e is not None:
            # exceptions raised by CNCalls are always Exception(str(clusterIndex))
            si = int(str(e))
            logger.warning("Failed to CNCalls for cluster n° %i, skipping it", si)
        else:
            counts2callsRes = futurecounts2callsRes.result()
            for exonIndex in range(len(counts2callsRes[2])):
                CNCallsArray[counts2callsRes[2][exonIndex], counts2callsRes[1]] = counts2callsRes[3][exonIndex]

            logger.info("Done copy number calls for cluster n°%i", counts2callsRes[0])

    # To be parallelised => browse clusters
    with ProcessPoolExecutor(paraClusters) as pool:
        for clustID in range(len(clusters)):
            #### validity sanity check
            if validity[clustID] == 0:
                logger.warning("cluster %s is invalid, low sample number", clustID)
                return
            
            ##### run prediction for current cluster
            futureRes = pool.submit(CNCalls.copyNumbersCalls.clusterCalls, clustID, exonsFPM,
                                    intergenicsFPM, samples, exons, clusters, ctrlsClusters,
                                    specClusters, CNTypes, exonsBool, outFile, samps2Check)

            futureRes.add_done_callback(mergeCalls)

    #####################################################
    # Print exon defs + calls to outFile
    callsFile = os.path.join(outFile, "exonCNCalls_" + str(len(samples)) + "samps_" + str(len(clusters)) + "clusters.tsv.gz")
    CNCalls.CNCallsFile.printCNCallsFile(CNCallsArray, exons, samples, callsFile)

    thisTime = time.time()
    logger.debug("Done printing calls for all (non-failed) clusters, in %.2fs", thisTime - startTime)
    logger.info("ALL DONE")


####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    scriptName = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(scriptName)

    try:
        main(sys.argv)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + scriptName + " : " + str(e) + "\n")
        sys.exit(1)
