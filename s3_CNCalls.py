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
    plotFolder = ""
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
printed in pdf files created in plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
            with '.gz', can have a path component but the subdir must exist
    --prevcncalls [str] optional: pre-existing copy number calls file produced by this program,
            possibly gzipped, the likelihoods of copy number are copied
            for samples contained in immutable clusters between old and new versions of the clustering files.
    --prevclusts [str] optional: pre-existing clustering file produced by s2_clusterSamps.py for the same
            timestamp as the pre-existing copy number call file.
    --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    --plotFolder [str]: subdir (created if needed) where result plots files will be produced, wish to monitor the filters
    --samps2Check [str]: comma-separated list of sample names, to plot calls, must be passed with --plotDir
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "out=", "prevcncalls=",
                                                           "prevclusts=", "jobs=", "plotDir=", "samps2Check="])
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
            prevClustsFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif (opt in ("--plotDir")):
            plotFolder = value
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

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other args
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

    if (plotFolder == "" and samps2Check != ""):
        raise Exception("you must use --checkSamps with --plotDir, not independently. Try " + scriptName + " --help")

    if plotFolder != "":
        # test plotdir last so we don't mkdir unless all other args are OK
        if not os.path.isdir(plotFolder):
            try:
                os.mkdir(plotFolder)
            except Exception as e:
                raise Exception("plotDir " + plotFolder + " doesn't exist and can't be mkdir'd: " + str(e))
        if samps2Check != "":
            samps2Check = samps2Check.split(",")

    # AOK, return everything that's needed
    return(countsFile, newClustsFile, outFile, prevCNCallsFile, prevClustsFile, jobs, plotFolder, samps2Check)


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
    (countsFile, clustsFile, outFile, prevCNCallsFile, prevClustsFile, jobs, plotFolder, samps2Check) = parseArgs(argv)

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
    
    # if CNCalls fails for any clusters, we have to remember their indexes
    # and only expunge them at the end -> save their indexes in failedClusters
    failedClusters = []

    # total number of samples that still need to be processed
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
        # parallel, each sample will be processed using coresPerSample.
        # -> we target targetCoresPerSample coresPerSample, this is increased if we
        #    have few samples to process (and we use ceil() so we may slighty overconsume)
        targetCoresPerCluster = 3
        paraClusters = min(math.ceil(jobs / targetCoresPerCluster), len(clusters))
        coresPerCluster = math.ceil(jobs / paraClusters)
        logger.info("%i new cluster => will process %i in parallel, using up to %i cores/cluster",
                    len(clusters), paraClusters, coresPerCluster)

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
                #  exceptions raised by CNCalls are always Exception(str(sampleIndex))
                si = int(str(e))
                logger.warning("Failed to CNCalls for cluster n° %s, skipping it", clusters[si])
                failedClusters.append(si)
            else:
                counts2callsRes = futurecounts2callsRes.result()
                si = counts2callsRes[0]
                for exonIndex in range(len(counts2callsRes[1])):
                    CNCallsArray[counts2callsRes[2][exonIndex], counts2callsRes[1]] = counts2callsRes[3][exonIndex]

                logger.info("Done copy number calls for cluster n°%s", counts2callsRes[0])

        # To be parallelised => browse clusters
        with ProcessPoolExecutor(paraClusters) as pool:
            for clustID in range(len(clusters)):
                logger.info("#### Process cluster n°%i", clustID)

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

                ##### run prediction for current cluster
                futureRes  = pool.submit(CNCalls.copyNumbersCalls.CNCalls(clustID, exonsFPM, intergenicsFPM, samples,
                                                                          exons, clusters, ctrlsClusters, sourceClusters,
                                                                          maskSourceExons, plotFolder, samps2Check))
                
                futureRes.add_done_callback(mergeCalls)

                thisTime = time.time()
                logger.debug("Done Copy Number Exons Calls for cluster n°%s, in %.2f s", clustID, thisTime - startTime)
                startTime = thisTime

        #####################################################
        # Print exon defs + calls to outFile
        CNCalls.CNCallsFile.printCNCallsFile(CNCallsArray, exons, clusters, outFile)

        thisTime = time.time()
        logger.debug("Done printing calls for all (non-failed) samples, in %.2fs", thisTime - startTime)
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
