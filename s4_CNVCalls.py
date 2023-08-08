###############################################################################################
################################ JACNEx  step 4:  CNV calling #################################
###############################################################################################
# Given three TSV files containing fragment counts, sample clusters, and distribution parameters
# for fitting the copy number profile, obtain a VCF file of copy number variations for all samples.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import math
import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import exonFilteringAndParams.exonParamsFile
import groupCalls.likelihoods
import groupCalls.transitions
import groupCalls.HMM

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
    paramsFile = ""
    outFile = ""
    # optionnal args with default values
    jobs = round(0.8 * len(os.sched_getaffinity(0)))
    plotDir = "./plotDir/"
    padding = 10
    BPDir = ""
    
    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Given three TSV files containing fragment counts, sample clusters, and distribution parameters
for fitting the copy number profile:
It calculates likelihoods for each sample and copy number, performs a hidden Markov chain
to obtain the best predictions, groups exons to form copy number variants (CNVs),
and deduces exact breakpoints if possible.
Finally, it generates a VCF (Variant Call Format) file for all analyzed samples.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
                    [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
                    File obtained from 2_clusterSamps.py.
    --params [str]: TSV file contains exon definitions in its first four columns,
                    followed by distribution parameters ["loc", "scale"] for exponential
                    and Gaussian distributions, and an additional column indicating the
                    exon filtering status for each cluster.
                    The file is generated using the 3_CNDistParams.py script.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
    --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    --plotDir[str]: sub-directory in which the graphical PDFs will be produced, default:  """ + plotDir + """
    --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
    --BPDir [str]: folder containing gzipped or ungzipped TSV for all samples analysed.
                   Files obtained from s1_countFrags.py
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "params=", "out=",
                                                           "jobs=", "plotDir=", "padding=", "BPDir="])
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
        elif (opt in ("--params")):
            paramsFile = value
        elif opt in ("--out"):
            outFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif (opt in ("--plotDir")):
            plotDir = value
        elif opt in ("--padding"):
            padding = value
        elif (opt in ("--BPDir")):
            BPDir = value
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

    if paramsFile == "":
        raise Exception("you must provide a continuous distribution parameters file use --params. Try " + scriptName + " --help.")
    elif (not os.path.isfile(paramsFile)):
        raise Exception("paramsFile " + paramsFile + " doesn't exist.")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

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

    if os.path.isdir(BPDir):
        print("TODO dev BPFolder treatments")

    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))
    
    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, clustsFile, paramsFile, outFile, jobs, plotDir, padding, BPDir)


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
    (countsFile, clustsFile, paramsFile, outFile, jobs, plotDir, padding, BPDir) = parseArgs(argv)

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

    ####################
    # parse params clusters (parameters of continuous distributions fitted on CN0, CN2 coverage profil)
    try:
        (exParams, exp_loc, exp_scale, paramsTitles) = exonFilteringAndParams.exonParamsFile.parseExonParamsFile(paramsFile, len(exons), len(clust2samps))
    except Exception as e:
        raise Exception("parseParamsFile failed for %s : %s", paramsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing paramsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # calculate likelihoods
    # cette étape peut être multiprocessée
    #######################
    CNStatus = ["CN0", "CN1", "CN2", "CN3"]
    # likelihood matrix creation
    likelihoodsArray = groupCalls.likelihoods.allocateLikelihoodsArray(len(samples), len(exons), len(CNStatus))

    # decide how the work will be parallelized
    # we are allowed to use jobs cores in total: we will process paraClusters clusters in
    # parallel
    # -> we target targetCoresPerCluster, this is increased if we
    #    have few clusters to process (and we use ceil() so we may slighty overconsume)
    paraClusters = min(math.ceil(jobs / 2), len(clust2samps))
    logger.info("%i new clusters => will process %i in parallel", len(clust2samps), paraClusters)

    #####################################################
    # mergeLikelihoods:
    # arg: a Future object returned by ProcessPoolExecutor.submit(groupCalls.likelihoods.observationCounts2Likelihoods).
    # observationCounts2Likelihoods returns a 4-element tuple (clusterID, relevantCols, relevantRows, likelihoodArray).
    # If something went wrong, raise log;
    # otherwise fill column at index relevantCols and row at index relevantRows in likelihoodsArray
    # with likelihoods stored in likelihoodsArray
    def mergeLikelihoods(futurecounts2Likelihoods):
        e = futurecounts2Likelihoods.exception()
        if e is not None:
            # exceptions raised by observationCounts2Likelihoods are always Exception(str(clusterIndex))
            logger.warning("Failed to observationCounts2Likelihoods for cluster n° %s, skipping it", str(e))
        else:
            counts2LikelihoodsRes = futurecounts2Likelihoods.result()
            for exonIndex in range(len(counts2LikelihoodsRes[2])):
                likelihoodsArray[counts2LikelihoodsRes[2][exonIndex], counts2LikelihoodsRes[1]] = counts2LikelihoodsRes[3][exonIndex]
            logger.info("Likelihoods calculated for cluster n°%s, NbOfSampsFilled %i, NbOfExonCalls %i/%i",
                        counts2LikelihoodsRes[0], len(counts2LikelihoodsRes[1])//len(CNStatus),
                        len(counts2LikelihoodsRes[2]), len(exons))

    # To be parallelised => browse clusters
    with ProcessPoolExecutor(paraClusters) as pool:
        for clusterID in clust2samps.keys():
            #### validity sanity check
            if not clustIsValid[clusterID]:
                logger.warning("cluster %s is invalid, low sample number", clusterID)
                continue

            ##### run prediction for current cluster
            futureRes = pool.submit(groupCalls.likelihoods.counts2Likelihoods, clusterID, samples, exonsFPM,
                                    clust2samps, exp_loc, exp_scale, exParams, len(CNStatus), len(paramsTitles))

            futureRes.add_done_callback(mergeLikelihoods)
    
    thisTime = time.time()
    logger.debug("Done calculate likelihoods, in %.2fs", thisTime - startTime)
    startTime = thisTime
    
    ####################
    # calculate the transition matrix from the likelihoods
    priors = np.array([6.34e-4, 2.11e-3, 9.96e-1, 1.25e-3])
    
    try:
        outPlotFile = os.path.join(plotDir, "CN_Frequencies_Likelihoods_Plot.pdf")
        transMatrix = groupCalls.transitions.getTransMatrix(likelihoodsArray, priors, samples, CNStatus, outPlotFile)
    except Exception as e:
        logger.error("getTransMatrix failed : %s", repr(e))
        raise Exception("getTransMatrix failed")

    thisTime = time.time()
    logger.debug("Done getTransMatrix, in %.2fs", thisTime - startTime)
    startTime = thisTime

    logger.error("EARLY EXIT, working on assignGender for now")
    return()
    # ####################
    # # apply HMM and obtain CNVs

    # # --CN [str]: TXT file contains two lines separated by tabulations.
    # # The first line consists of entries naming the default copy number status,
    # # default: """ + CNStatus + """, and the second line contains the prior
    # # probabilities of occurrence for the events, default: """ + priors + """
    # # obtained from 1000 genome data (doi:10.1186/1471-2105-13-305).
    # #CNStatus = ["CN0", "CN1", "CN2", "CN3"]
    
    # #############
    # # CNVs calls
    # try:
    #     for sampIndex in range(len(samples)):
    #         # Extract the CN calls for the current sample
    #         CNcallOneSamp = likelihoodsArray[:, sampIndex * len(CNStatus): sampIndex * len(CNStatus) + len(CNStatus)]
    #         try:
    #             # Perform CNV inference using HMM
    #             CNVsSampList = groupCalls.HMM.viterbi(CNcallOneSamp, transMatrix)
    #         except Exception as e:
    #             logger.error("viterbi failed : %s", repr(e))
    #             raise Exception("viterbi failed")

    #         # Create a column with sampIndex values
    #         sampIndexColumn = np.full((CNVsSampList.shape[0], 1), sampIndex, dtype=CNVsSampList.dtype)
    #         # Concatenate CNVsSampList with sampIndex column
    #         CNVsSampListAddSampInd = np.hstack((CNVsSampList, sampIndexColumn))
    #         # Stack CNVsSampList_with_sampIndex vertically with previous results
    #         CNVArray = np.vstack((CNVArray, CNVsSampListAddSampInd))

    # except Exception as e:
    #     logger.error("CNVs calls failed : %s", repr(e))
    #     raise Exception("CNVs calls failed")

    # thisTime = time.time()
    # logger.debug("Done CNVs calls, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # ############
    # # vcf format
    # try:
    #     resVcf = groupCalls.HMM.CNV2Vcf(CNVArray, exons, samples, padding)
    # except Exception as e:
    #     logger.error("CNV2Vcf failed : %s", repr(e))
    #     raise Exception("CNV2Vcf failed")

    # thisTime = time.time()
    # logger.debug("Done CNV2Vcf, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # ###########
    # # print results
    # try:
    #     groupCalls.HMM.printVcf(resVcf, outFile, scriptName, samples)
    # except Exception as e:
    #     logger.error("printVcf failed : %s", repr(e))
    #     raise Exception("printVcf failed")

    # thisTime = time.time()
    # logger.debug("Done printVcf, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # thisTime = time.time()
    # logger.debug("Done printing groupingCalls, in %.2fs", thisTime - startTime)
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
