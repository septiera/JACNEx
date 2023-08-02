###############################################################################################
######################################## JACNEx step 3: Copy numbers calls ##################
###############################################################################################
# Given a TSV of exon fragment counts produced by 1_countFrags.py
# and a TSV with clustering information produced by 2_clusterSamps.py:
# calculates robust Gaussian distribution parameters for sample exon counts and
# exponential distribution parameters for intergenic regions counts within a patient cluster.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import math
import time
import logging
from concurrent.futures import ProcessPoolExecutor

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import clusterSamps.genderPrediction
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
# If anything is wrong, raise Exception("EXPLICIT ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    clustFile = ""
    outFile = ""
    # jobs default: 80% of available cores
    plotDir = "./plotDir/"
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Processes two TSV files: one containing exon fragment counts and another with samples clusters.
It filters out non-callable exons (with no coverage) and computes parameters for two distributions:
an exponential distribution (loc=0, scale=lambda) for CN0 and a robust Gaussian distribution 
(loc=mean, scale=stdev) for CN2.
The results are displayed in TSV format on the standard output (stdout).
The first row represents the parameters of the exponential distribution, while the subsequent rows
contain the exon definitions along with the corresponding 'loc' and 'scale' parameters for the
Gaussian distribution.
In addition, all the graphics (exponential fit on count data and dendograms summarising the
proportions of filtered and unfiltered exons) are printed in pdf files created in plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
            with '.gz', can have a path component but the subdir must exist
    --plotDir[str]: sub-directory in which the graphical PDFs will be produced, default:  """ + plotDir + """
    --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "out=", "plotDir=", "jobs="])
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
        elif (opt in ("--plotDir")):
            plotDir = value
        elif opt in ("--jobs"):
            jobs = value
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
    
    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other args 
    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))
    
    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, clustFile, outFile, plotDir, jobs)


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
    (countsFile, clustsFile, outFile, plotDir, jobs) = parseArgs(argv)

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

    ###################
    # parse clusters informations
    try:
        (clust2samps, samp2clusts, fitWith, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime
    
    # warning: there are no checks performed on the number of samples in 'samples' and
    # the samples contained in clusters. However, if a sample is present in a cluster
    # but not in 'samples', Python will raise an error when selecting the non-existent
    # counts column. On the other hand, if an additional sample is present, it will not
    # be analyzed and will be ignored without raising an error.
    # In pipeline mode, this should not happen.
    
    ####################
    # CN Calls
    # calculates distribution parameters:
    # -fits an exponential distribution to the entire dataset of intergenic counts (CN0),
    # -selects cluster-specific exons on autosomes, chrY, or chrX.
    #  Uninterpretable exons are filtered out while robustly calculating parameters for
    #  a fitted Gaussian distribution.
    
    try:
        (expon_loc, exp_scale, unCaptFPMLimit) = CNCalls.copyNumbersCalls.CN0ParamsAndFPMLimit(intergenicsFPM, plotDir)
    except Exception as e:
        raise Exception("CN0ParamsAndFPMLimit failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done parsing CN0ParamsAndFPMLimit, in %.2f s", thisTime - startTime)
    startTime = thisTime
    
    # ######################################################
    # # !!! (samples to be changed with the comparision between previous and current analysis)
    # paramsToKeep = ["loc", "scale"]
    # paramsArray = CNCalls.CNCallsFile.allocateParamsArray(len(exons), len(clusters), len(paramsToKeep))

    # ####################################################
    # # CN Calls
    # ##############################
    # # decide how the work will be parallelized
    # # we are allowed to use jobs cores in total: we will process paraClusters clusters in
    # # parallel
    # # -> we target targetCoresPerCluster, this is increased if we
    # #    have few clusters to process (and we use ceil() so we may slighty overconsume)
    # paraClusters = min(math.ceil(jobs / 2), len(clusters))
    # logger.info("%i new clusters => will process %i in parallel", len(clusters), paraClusters)

    # # identifying autosomes and gonosomes "exons" index
    # # recall clusters are derived from autosomal or gonosomal analyses
    # try:
    #     exonsBool = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)
    # except Exception as e:
    #     raise Exception("getSexChrIndexes failed %s", e)

    # #####################################################
    # # Define nested callback for processing CNCalls() result (so CNCallsArray et al
    # # are in its scope)

    # # mergeCalls:
    # # arg: a Future object returned by ProcessPoolExecutor.submit(CNCalls.copyNumbersCalls.CNCalls).
    # # CNCalls() returns a 4-element tuple (clustID, colIndInCNCallsArray, sourceExons, clusterCalls).
    # # If something went wrong, log and populate failedClusters;
    # # otherwise fill column at index colIndInCNCallsArray and row at index sourceExons in CNCallsArray
    # # with calls stored in clusterCalls
    # def mergeCalls(futurecounts2callsRes):
    #     e = futurecounts2callsRes.exception()
    #     if e is not None:
    #         # exceptions raised by CNCalls are always Exception(str(clusterIndex))
    #         logger.warning("Failed to CNCalls for cluster n° %s, skipping it", str(e))
    #     else:
    #         counts2callsRes = futurecounts2callsRes.result()
    #         for exonIndex in range(len(counts2callsRes[2])):
    #             paramsArray[counts2callsRes[2][exonIndex], counts2callsRes[1]] = counts2callsRes[3][exonIndex]

    #         logger.info("Done copy number calls for cluster n°%i", counts2callsRes[0])

    # # To be parallelised => browse clusters
    # with ProcessPoolExecutor(paraClusters) as pool:
    #     for clustID in range(len(clusters)):
    #         #### validity sanity check
    #         if validity[clustID] == 0:
    #             logger.warning("cluster %s is invalid, low sample number", clustID)
    #             continue

    #         ##### run prediction for current cluster
    #         futureRes = pool.submit(CNCalls.copyNumbersCalls.clusterCalls, clustID, exonsFPM,
    #                                 intergenicsFPM, exons, clusters, ctrlsClusters, specClusters,
    #                                 exonsBool, outFile, paramsToKeep)

    #         futureRes.add_done_callback(mergeCalls)

    # #####################################################
    # # Print exon defs + calls to outFile
    # CNCalls.CNCallsFile.printCNCallsFile(paramsArray, exons, clusters, paramsToKeep, outFile)

    # thisTime = time.time()
    # logger.debug("Done printing calls for all (non-failed) clusters, in %.2fs", thisTime - startTime)
    # logger.info("ALL DONE")


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
