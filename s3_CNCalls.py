###############################################################################################
######################## JACNEx step 3: exon filtering and calling ############################
###############################################################################################
# Given a TSV of exon fragment counts produced by 1_countFrags.py
# and a TSV with clustering information produced by 2_clusterSamps.py:
# It filters out non-callable exons and computes parameters for two distributions:
# an exponential distribution (loc=0, scale=lambda) for CN0 and a robust Gaussian distribution
# (loc=mean, scale=stdev) for CN2.
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
import clusterSamps.genderPrediction
import CNCalls.CNCallsFile
import CNCalls.copyNumbersCalls
import figures.plots

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
    # calculates distribution parameters:
    # 1) fits an exponential distribution to the entire dataset of intergenic counts (CN0)
    try:
        (expon_loc, exp_scale, unCaptFPMLimit) = CNCalls.copyNumbersCalls.CN0ParamsAndFPMLimit(intergenicsFPM, plotDir)
    except Exception as e:
        raise Exception("CN0ParamsAndFPMLimit failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done CN0ParamsAndFPMLimit, loc=%.2f, scale=%.2f, in %.2f s",expon_loc, exp_scale, thisTime - startTime)
    startTime = thisTime

    ###
    # 2) selects cluster-specific exons on autosomes, chrY, or chrX.
    # Uninterpretable exons are filtered out while robustly calculating parameters for
    # a fitted Gaussian distribution.

    exonOnSexChr = clusterSamps.genderPrediction.exonOnSexChr(exons)

    # defined the result columns for each cluster.
    # 'loc' column corresponds to the mean of the Gaussian distribution
    # 'scale' column represents the standard deviation of the Gaussian distribution
    # 'filterStatus' column indicates the status of the corresponding exon
    expectedColNames = ["loc", "scale", "filterStatus"]

    # Output matrix creation
    CN2ParamsArray = CNCalls.CNCallsFile.allocateParamsArray(len(exons), len(clust2samps), len(expectedColNames))
    logger.info(CN2ParamsArray.shape)
    # filterStatus represents the set of filters applied to the exons.
    # The indexes of this list will be used to map the filtered status of each
    # exon in the output file.
    # e.g: index 0 corresponds to an exon filtered as 'notCaptured',
    # while the index -1 (initialized during the output matrix allocation)
    # represents exons that are not used for cluster parameter calculation.
    filterStatus = ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"]

    # generates a PDF file containing pie charts summarizing the filters 
    # applied to exons in each cluster.
    # matplotlib.backends.backend_pdf.PdfPages object matplotOpenFile is used to
    # open the PDF file and add the pie charts to it.
    namePieChartsFile = "exonsFiltersSummary_pieChart_" + str(len(clust2samps)) + "Clusters.pdf"
    pdfPieCharts = os.path.join(plotDir, namePieChartsFile)
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdfPieCharts)

    #######################
    # decide how the work will be parallelized
    # we are allowed to use jobs cores in total: we will process paraClusters clusters in
    # parallel
    # -> we target targetCoresPerCluster, this is increased if we
    #    have few clusters to process (and we use ceil() so we may slighty overconsume)
    paraClusters = min(math.ceil(jobs / 2), len(clust2samps))
    logger.info("%i new clusters => will process %i in parallel", len(clust2samps), paraClusters)

    #####################################################
    # mergeParams:
    # arg: a Future object returned by ProcessPoolExecutor.submit(CNCalls.copyNumbersCalls.exonFilterAndCN2Params).
    # exonFilterAndCN2Params() returns a 4-element tuple (clustID, colIndInParamsArray, sourceExons, clusterCN2Params).
    # If something went wrong, log and populate failedClusters;
    # otherwise fill column at index colIndInParamsArray and row at index sourceExons in paramsArray
    # with calls stored in clusterCN2Params
    def mergeParams(futurecounts2ParamsRes):
        e = futurecounts2ParamsRes.exception()
        if e is not None:
            # exceptions raised by ExonFilterAndCN2Params are always Exception(str(clusterIndex))
            logger.warning("Failed to ExonFilterAndCN2Params for cluster n° %s, skipping it", str(e))
        else:
            counts2ParamsRes = futurecounts2ParamsRes.result()
            for exonIndex in range(len(counts2ParamsRes[2])):
                CN2ParamsArray[counts2ParamsRes[2][exonIndex], counts2ParamsRes[1]] = counts2ParamsRes[3][exonIndex]
                
            try:
                # Generate pie chart for the cluster based on the filterStatus
                exStatusArray = counts2ParamsRes[3][:, expectedColNames.index("filterStatus")]
                figures.plots.plotPieChart(counts2ParamsRes[0], filterStatus, exStatusArray, matplotOpenFile)
            except Exception as e:
                logger.error("plotPieChart failed for cluster %s : %s", clustID, repr(e))
                raise

            logger.info("Done copy number calls for cluster n°%s", counts2ParamsRes[0])

    # To be parallelised => browse clusters
    with ProcessPoolExecutor(paraClusters) as pool:
        for clustID in clust2samps.keys():
            #### validity sanity check
            if not clustIsValid[clustID]:
                logger.warning("cluster %s is invalid, low sample number", clustID)
                continue

            ##### run prediction for current cluster
            futureRes = pool.submit(CNCalls.copyNumbersCalls.exonFilterAndCN2Params, clustID,
                                    exonsFPM, samples, clust2samps, fitWith, exonOnSexChr,
                                    unCaptFPMLimit, expectedColNames, filterStatus)

            futureRes.add_done_callback(mergeParams)

    # close PDF file containing pie charts
    matplotOpenFile.close()

    # #####################################################
    # # Print exon defs + calls to outFile
    # CNCalls.CNCallsFile.printCNCallsFile(CN2ParamsArray, exons, clusters, paramsToKeep, outFile)

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
