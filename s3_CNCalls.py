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
    prevClustsFile = ""
    outFolder = ""
    samps2Check = ""
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts and a TSV with clustering information,
deduces the copy numbers (CN) observation probabilities (likelihoods), per exon and for each sample.
Results are printed to stdout in TSV format (possibly gzipped): first 4 columns hold the exon
definitions padded and sorted, subsequent columns (four per sample, in order CN0,CN2,CN2,CN3+)
hold the likelihoods.
If a pre-existing copy number calls file (with --cncalls) produced by this program associated with
a previous clustering file are provided (with --prevclusts), extraction of the likelihoods
for the samples in homogeneous clusters between the two versions, otherwise the copy number calls is performed.
In addition, all graphical support (pie chart of exon filtering per cluster) are
printed in pdf files created in outDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --outDir [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
            with '.gz', can have a path component but the subdir must exist
    --prevcncalls [str] optional: pre-existing copy number calls file produced by this program,
            possibly gzipped, the likelihoods of copy number are copied
            for samples contained in immutable clusters between old and new versions of the clustering files.
    --prevclusts [str] optional: pre-existing clustering file produced by s2_clusterSamps.py for the same
            timestamp as the pre-existing copy number call file.
    --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    --samps2Check [str]: comma-separated list of sample names, to plot calls, must be passed with --outDir
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "outDir=", "prevcncalls=",
                                                           "prevclusts=", "jobs=", "samps2Check="])
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
        elif opt in ("--outDir"):
            outFolder = value
        elif (opt in ("--prevcncalls")):
            prevCNCallsFile = value
        elif (opt in ("--prevclusts")):
            prevClustsFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif (opt in ("--samps2Check")):
            samps2Check = value
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

    if outFolder != "":
        if not os.path.isdir(outFolder):
            try:
                os.mkdir(outFolder)
            except Exception as e:
                raise Exception("outDir " + outFolder + " doesn't exist and can't be mkdir'd: " + str(e))

    #####################################################
    # Check other argsjobs = round(0.8 * len(os.sched_getaffinity(0)))
    if (prevCNCallsFile != "" and prevClustsFile == "") or (prevCNCallsFile == "" and prevClustsFile != ""):
        raise Exception("you should not use --cncalls and --prevclusts alone but together. Try " + scriptName + " --help")

    if (prevCNCallsFile != "") and (not os.path.isfile(prevCNCallsFile)):
        raise Exception("CNCallsFile " + prevCNCallsFile + " doesn't exist")

    if (prevClustsFile != "") and (not os.path.isfile(prevClustsFile)):
        raise Exception("previous clustering File " + prevClustsFile + " doesn't exist")

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        if samps2Check != "":
            samps2Check = samps2Check.split(",")
    else:
        if samps2Check != "":
            raise Exception("samps2Check should not be provided when the logger level is not set to DEBUG.")

    # AOK, return everything that's needed
    return(countsFile, newClustsFile, outFolder, prevCNCallsFile, prevClustsFile, jobs, samps2Check)


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
    (countsFile, clustsFile, outFolder, prevCNCallsFile, prevClustsFile, jobs, samps2Check) = parseArgs(argv)

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
        (clusters, ctrlsClusters, validity, sourceClusters) = clusterSamps.clustFile.parseClustsFile(clustsFile, samples)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ######################################################
    # allocate CNcallsArray, and populate it with pre-calculated observed probabilities
    # if CNCallsFile and prevClustFile are provided.
    # also returns a boolean np.array to identify the clusters to be reanalysed if the clusters change
    try:
        (CNCallsArray, callsFilled) = CNCalls.CNCallsFile.extractCNCallsFromPrev(exons, samples, clusters, prevCNCallsFile, prevClustsFile)
    except Exception as e:
        raise Exception("extractCNCallsFromPrev failed - " + str(e))

    thisTime = time.time()
    logger.debug("Done parsing previous CNCallsFile and prevClustFile, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # total number of clusters that still need to be processed
    nbOfClustersToProcess = len(clusters)
    for clustIndex in range(len(clusters)):
        if callsFilled[clustIndex]:
            nbOfClustersToProcess -= 1

    if nbOfClustersToProcess == 0:
        logger.info("all provided clusters are identical to the previous callsFile, not producing a new one")
    else:
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
            maskSourceExons = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)
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
                for exonIndex in range(len(counts2callsRes[1])):
                    CNCallsArray[counts2callsRes[2][exonIndex], counts2callsRes[1]] = counts2callsRes[3][exonIndex]

                logger.info("Done copy number calls for cluster n°%i", counts2callsRes[0])

        # To be parallelised => browse clusters
        with ProcessPoolExecutor(paraClusters) as pool:
            for clustID in range(len(clusters)):

                ##### validity sanity check
                if validity[clustID] == 0:
                    logger.warning("cluster %s is invalid, low sample number", clustID)
                    continue

                ##### previous data sanity filters
                # test if the cluster has already been analysed
                # and the filling of CNcallsArray has been done
                if callsFilled[clustID]:
                    logger.info("samples in cluster %s, already filled from prevCallsFile", clustID)
                    continue

                # extracts and appends indices of samples to plot
                clusterSamps2Plot = []
                if samps2Check:
                    for i in range(len(samples)):
                        if ((samples[i] in samps2Check) and (i in clusters[clustID])):
                            clusterSamps2Plot.extend(i)

                ##### run prediction for current cluster
                futureRes = pool.submit(CNCalls.copyNumbersCalls.CNCalls, clustID, exonsFPM,
                                        intergenicsFPM, samples, exons, clusters, ctrlsClusters,
                                        sourceClusters, maskSourceExons, outFolder, clusterSamps2Plot)

                futureRes.add_done_callback(mergeCalls)

        #####################################################
        # Print exon defs + calls to outFile
        callsFile = os.path.join(outFolder, "exonCNCalls_" + str(len(samples)) + "samps_" + str(len(clusters)) + "clusters.tsv")
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
