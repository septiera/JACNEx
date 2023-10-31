###############################################################################################
######################## JACNEx step 3: exon filtering and calling ############################
###############################################################################################
# Given a TSV of exon fragment counts produced by 1_countFrags.py
# and a TSV with clustering information produced by 2_clusterSamps.py:
# It filters out non-callable exons and computes parameters for two distributions:
# an exponential distribution (loc=0, scale=lambda) for CN0 and a Gaussian distribution
# (loc=mean, scale=stdev) for CN2 distinguishing autosomes and gonosomes.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import math
import time
import logging
import matplotlib.backends.backend_pdf
from concurrent.futures import ProcessPoolExecutor

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import exonCalls.exonCallsFile
import exonCalls.exonProcessing
import figures.plots

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
    clustsFile = ""
    outFile = ""
    # jobs default: 80% of available cores
    plotDir = "./plotDir/"
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given two TSV files: one containing the number of exon fragments and the other the sample clusters,
we filter out non-callable exons based on various criteria and compute parameters for two distributions:
1)- An exponential distribution (loc=0, scale=1/λ) derived from the distribution of normally uncovered
 intergenic regions to approximate the profile of homozygous deletions (CN0).
2)- A Gaussian distribution (loc=mean, scale=stdev) based on specific counts for each cluster to deduce
 a profile close to what is expected in the normal haploid state (CN2).
The results are saved in TSV format on the standard output (stdout) for each cluster chromosome type.
The first line (excluding the header) represents the parameters of the exponential distribution.
Subsequent lines contain exon definitions [CHR, START, END, EXONID], along with the corresponding
"loc" and "scale" parameters for the Gaussian distribution for each cluster.
Additionally, all circular diagrams summarizing the proportions of filtered and unfiltered exons are
generated and saved as PDF files in the plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 4 columns hold the sample cluster definitions.
                    [CLUSTER_ID, SAMPLES, FIT_WITH, VALID]. File obtained from 2_clusterSamps.py.
    --out [str]: Specify the output file where results will be saved. If the filename ends with '.gz,'
                 it will be compressed. You can include a path component, but ensure the specified
                 directory already exists. The final filename will be determined based on the chromosomes
                 used for cluster generation, potentially modifying the original filename in accordance
                 with the data processing logic.
    --plotDir [str]: sub-directory in which the graphical PDFs will be produced, default:  """ + plotDir + """
    --jobs [int]: cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
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
            clustsFile = value
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

    if clustsFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(clustsFile)):
        raise Exception("clustsFile " + clustsFile + " doesn't exist.")

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
    return(countsFile, clustsFile, outFile, plotDir, jobs)


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
    # parse and FPM-normalize the counts, distinguishing between exons and intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done parsing and normalizing countsFile, in %.2f s", thisTime - startTime)
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

    ####################
    # Exponential fitting = CN0 on intergenic counts.
    # Extracts distribution parameters and a threshold (uncaptThreshold)
    try:
        (expon_loc, expon_scale, uncaptThreshold) = exonCalls.exonProcessing.computeCN0Params(intergenicFPMs)
    except Exception as e:
        raise Exception("computeCN0Params failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done computeCN0Params, loc=%.2f, scale=%.2f, uncapThreshold=%.2f in %.2f s",
                 expon_loc, expon_scale, uncaptThreshold, thisTime - startTime)
    startTime = thisTime

    ####################
    # Exon Metrics Processing and Filtering
    ####################
    # Initialized a list[strs] for output metric names:
    # 'loc': the mean of the Gaussian distribution
    # 'scale': the standard deviation of the Gaussian distribution
    # 'filterStatus': filtering state indexes
    metricsNames = ["loc", "scale", "filterStates"]

    # output dictionaries: keys == clusterID and values == np.ndarray[floats],
    # dim = NBOfExons * NBOfMetrics, contains the fitting results of the Gaussian
    # distribution and filters for all exons and clusters.
    exonMetrics_A = {}
    exonMetrics_G = {}

    # filterStates list[strs] represents the set of filter names applied to the exons.
    # The indexes of this list will be used to map the filtered states of each
    # exon in the output file.
    # e.g: index 0 corresponds to an exon filtered as 'notCaptured',
    # while the index -1 (initialized during the output matrix allocation)
    # represents exons that are not used for cluster parameter calculation.
    filterStates = ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"]

    # generates a PDF file containing pie charts summarizing the filters
    # applied to exons in each cluster.
    # matplotlib.backends.backend_pdf.PdfPages object matplotOpenFile is used to
    # open the PDF file and add the pie charts to it.
    namePieChartsFile = "exonsFiltersSummary_pieChart_" + str(len(clust2samps)) + "Clusters.pdf"
    pdfPieCharts = os.path.join(plotDir, namePieChartsFile)
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdfPieCharts)

    # This step is parallelized across clusters,
    paraClusters = min(math.ceil(jobs / 2), len(clust2samps))
    logger.info("%i new clusters => will process %i in parallel", len(clust2samps), paraClusters)

    ############
    # updateMetricsDictionary:
    # arg: a Future object returned by ProcessPoolExecutor.submit(exonCalls.exonProcessing.processExonsAndComputeCN2Params).
    # returns a 3-element tupple (chromType, clusterID, clustExMetrics).
    # If something went wrong, raise error in log;
    # Otherwise, fill the exonMetrics key with the clusterID and the corresponding chromType,
    # and associate it with the clustExMetrics np.ndarray dim=NbOfExons * NbOfMetrics
    # In addition, plot a pie chart summarizing the performed filtering for the current cluster.
    def updateMetricsDictionary(futurecounts2ParamsRes):
        e = futurecounts2ParamsRes.exception()
        if e is not None:
            logger.warning("Failed to processExonsAndComputeCN2Params for cluster %s, skipping it", str(e))
        else:
            counts2ParamsRes = futurecounts2ParamsRes.result()

            if counts2ParamsRes[0] == "A":
                exonMetrics_A[counts2ParamsRes[1]] = counts2ParamsRes[2]
            else:
                exonMetrics_G[counts2ParamsRes[1]] = counts2ParamsRes[2]

            # Generate pie chart for the cluster based on the filterStatus
            try:
                figures.plots.plotPieChart(counts2ParamsRes[1], filterStates,
                                           counts2ParamsRes[2][:, metricsNames.index("filterStates")],
                                           matplotOpenFile)
            except Exception as e:
                logger.error("plotPieChart failed for cluster %s : %s", counts2ParamsRes[1], repr(e))
                raise

            logger.info("Done adjustExonMetricsWithFilters for cluster %s", counts2ParamsRes[1])

    # To be parallelised => browse clusters
    with ProcessPoolExecutor(paraClusters) as pool:
        for clusterID in clust2samps.keys():
            #### validity sanity check
            if not clustIsValid[clusterID]:
                logger.warning("cluster %s is invalid, low sample number", clusterID)
                continue

            ### chromType [str]: variable distinguishes between analyses of
            # sex chromosomes (gonosomes) and non-sex chromosomes (autosomes).
            # autosomes
            if clusterID.startswith("A"):
                chromType = "A"
                futureRes = pool.submit(exonCalls.exonProcessing.processExonsAndComputeCN2Params,
                                        clusterID, chromType, autosomeFPMs, samples, clust2samps,
                                        fitWith, uncaptThreshold, metricsNames, filterStates)
            # gonosomes
            elif clusterID.startswith("G"):
                chromType = "G"
                futureRes = pool.submit(exonCalls.exonProcessing.processExonsAndComputeCN2Params,
                                        clusterID, chromType, gonosomeFPMs, samples, clust2samps,
                                        fitWith, uncaptThreshold, metricsNames, filterStates)

            else:
                logger.error("Cluster %s doesn't distinguish gonosomal from autosomal analyses.", clusterID)
                raise

            futureRes.add_done_callback(updateMetricsDictionary)

    # close PDF file containing pie charts
    matplotOpenFile.close()

    #####################################################
    # Print exon defs + metrics to outFile
    # requires two independent outputs, one for the gonosomes and the other for the autosomes
    outputName = os.path.splitext(outFile)[0]
    # autosomes
    exonCalls.exonCallsFile.printParamsFile(outputName + "_A.gz", exonMetrics_A, metricsNames,
                                            expon_loc, expon_scale, autosomeExons)
    # gonosomes
    exonCalls.exonCallsFile.printParamsFile(outputName + "_G.gz", exonMetrics_G, metricsNames,
                                            expon_loc, expon_scale, gonosomeExons)

    thisTime = time.time()
    logger.debug("Done printing exon metrics for all (non-failed) clusters, in %.2fs", thisTime - startTime)
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
