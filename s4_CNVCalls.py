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
import time
import logging
import numpy as np

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import exonFilteringAndParams.exonParamsFile
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
    BPFolder = ""
    padding = 10

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
    --BPFolder [str]: folder containing gzipped or ungzipped TSV for all samples analysed.
                      Files obtained from s1_countFrags.py
    --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "params=", "out=",
                                                           "BPFolder=", "padding="])
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
        elif (opt in ("--BPFolder")):
            BPFolder = value
        elif opt in ("--padding"):
            padding = value
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

    if os.path.isdir(BPFolder):
        print("TODO dev BPFolder treatments")

    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))

    # AOK, return everything that's needed
    return(countsFile, clustsFile, paramsFile, outFile, BPFolder, padding)


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
    (countsFile, clustsFile, paramsFile, outFile, BPFolder, padding) = parseArgs(argv)

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
        (exParams, exp_loc, exp_scale) = exonFilteringAndParams.exonParamsFile.parseExonParamsFile(paramsFile)
    except Exception as e:
        raise Exception("parseParamsFile failed for %s : %s", paramsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing paramsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    # # --CN [str]: TXT file contains two lines separated by tabulations.
    # # The first line consists of entries naming the default copy number status,
    # # default: """ + CNStatus + """, and the second line contains the prior
    # # probabilities of occurrence for the events, default: """ + priors + """
    # # obtained from 1000 genome data (doi:10.1186/1471-2105-13-305).
    # CNStatus = ["CN0", "CN1", "CN2", "CN3"]
    # priors = np.array([6.34e-4, 2.11e-3, 9.96e-1, 1.25e-3])

    # # args seem OK, start working
    # logger.debug("called with: " + " ".join(argv[1:]))
    # logger.info("starting to work")
    # startTime = time.time()

    # #############
    # # parse calls
    # try:
    #     (exons, samples, CNCallsArray) = CNCalls.CNCallsFile.parseCNcallsFile(callsFile, len(CNStatus))
    # except Exception as e:
    #     logger.error("parseCNcallsFile failed for %s : %s", callsFile, repr(e))
    #     raise Exception("parseCNcallsFile failed")

    # thisTime = time.time()
    # logger.debug("Done parse callsFile, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # ############
    # # obtaining a list of observations and a transition matrix based on the data.
    # try:
    #     transMatrix = groupCalls.transitions.getTransMatrix(CNCallsArray, priors, samples, CNStatus, os.path.dirname(outFile))
    # except Exception as e:
    #     logger.error("getTransMatrix failed : %s", repr(e))
    #     raise Exception("getTransMatrix failed")

    # thisTime = time.time()
    # logger.debug("Done getTransMatrix, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # #############
    # # CNVs calls
    # try:
    #     for sampIndex in range(len(samples)):
    #         # Extract the CN calls for the current sample
    #         CNcallOneSamp = CNCallsArray[:, sampIndex * len(CNStatus): sampIndex * len(CNStatus) + len(CNStatus)]
    #         try:
    #             # Perform CNV inference using HMM
    #             CNVsSampList = groupCalls.HMM.inferCNVsUsingHMM(CNcallOneSamp, exons, transMatrix, priors)
    #         except Exception as e:
    #             logger.error("inferCNVsUsingHMM failed : %s", repr(e))
    #             raise Exception("inferCNVsUsingHMM failed")

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
