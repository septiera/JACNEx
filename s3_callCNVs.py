###############################################################################################
######################## JACNEx step 3: exon filtering and calling ############################
###############################################################################################
# Given a TSV of exon fragment counts produced by 1_countFrags.py
# and a TSV with clustering information produced by 2_clusterSamps.py:
# The script facilitates calling CNVs and generates a VCF file with the detections.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import time
import logging

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import callCNVs.exonProcessing
import callCNVs.likelihoods

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
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given input from a fragment counting TSV file per sample and a TSV file
containing information about sample clustering.
Executes multiple processing steps:
a) Normalizes fragment counts to FPMs.
b) Filters out non-interpretable exons based on various criteria.
c) Adjusts continuous distribution and extracts parameters associated with
   CN0 (half normal) and CN2 (Gaussian) profiles.
d) Calculates likelihoods for each CN and sample.
e) Corrects likelihoods based on the probability of belonging to a given CN.
f) Calculating a transition matrix based on the data.
g) Applies an Hidden Markov Model for call grouping.
h) Formats the calls into VCF.
i) Prints the VCF.
The code contains specific parts for debugging during development, but these
are intended to be removed in the final version.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 4 columns hold the sample cluster definitions.
                    [CLUSTER_ID, SAMPLES, FIT_WITH, VALID]. File obtained from 2_clusterSamps.py.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist.
    --jobs [int]: cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "out=", "jobs=",
                                                           "plotDir=", "printParams="])
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

    # AOK, return everything that's needed
    return(countsFile, clustsFile, outFile, jobs)


###############################################################################
############################ DEBUG FUNCTIONS ##################################
###############################################################################
# createDebugFolder
# Creates a folder for debug plots if the logging level is set to DEBUG.
#
# Args:
# - mainFolder (str): Path to the main folder.
# - name (str): folder name
#
# Returns:
# - str: Path to the created debug plots folder, or None if the folder is not created.
def createDebugFolder(mainFolder, dirName):
    plotDir = None
    # Checks the logging level
    if logger.level <= logging.DEBUG:
        plotDir = os.path.join(mainFolder, dirName)
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))
    return plotDir


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
    (countsFile, clustsFile, outFile, jobs) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    plotDir = createDebugFolder(os.path.dirname(outFile), "DEBUG_PLOTS_zscoreDynamic")  # to remove

    ###################
    # parse and FPM-normalize the counts, differentiating between exons on autosomes and gonosomes,
    # as well as intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.info("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # parse clusters informations
    try:
        (clust2samps, samp2clusts, fitWith, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parseClustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # Filters out non-callable exons and computes parameters for two distributions:
    # a half normal distribution (loc=0, scale=stdev) for CN0 and a Gaussian distribution
    # (loc=mean, scale=stdev) for CN2 distinguishing autosomes and gonosomes.
    try:
        (hnorm_loc, hnorm_scale, uncaptThreshold) = callCNVs.exonProcessing.computeCN0Params(intergenicFPMs)
    except Exception as e:
        raise Exception("computeCN0Params failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done computeCN0Params, loc=%.2f, scale=%.2f, uncaptThreshold=%.2f in %.2f s",
                 hnorm_loc, hnorm_scale, uncaptThreshold, thisTime - startTime)
    startTime = thisTime

    try:
        speDir = createDebugFolder(plotDir, "exonFilteringSummary")  # to remove
        (CN2Params_A, CN2Params_G) = callCNVs.exonProcessing.parallelClusterProcessing(autosomeFPMs, gonosomeFPMs, samples,
                                                                                       uncaptThreshold, clust2samps, fitWith,
                                                                                       clustIsValid, speDir, jobs)
    except Exception as e:
        raise Exception("parallelClusterProcessing failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done parallelClusterProcessing in %.2f s", thisTime - startTime)
    startTime = thisTime

    # - States to be emitted by the HMM corresponding to different types of copy numbers:
    # CNO = homodeletion , CN1 = heterodeletion, CN2 = diploid (normal copy number)
    # CN3 = duplication, we decided not to distinguish the number of copies, so 3+.
    CNStates = ["CN0", "CN1", "CN2", "CN3"]

    # - CNState occurrence probabilities of the human genome, obtained from 1000 genome data
    # (doi:10.1186/1471-2105-13-305).
    priors = [6.34e-4, 2.11e-3, 9.96e-1, 1.25e-3]

    ####################
    # Likelihood calculation for each sample (pseudo emission array)
    try:
        (likelihoods_A, likelihoods_G) = callCNVs.likelihoods.allChrom(samples, autosomeFPMs, gonosomeFPMs,
                                                                       clust2samps, fitWith, hnorm_loc,
                                                                       hnorm_scale, CN2Params_A, CN2Params_G,
                                                                       CNStates, priors, jobs, speDir)
    except Exception as e:
        raise Exception("likelihoods.allChrom failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done likelihoods.allChrom in %.2f s", thisTime - startTime)
    startTime = thisTime

    #########
    # generates a transition matrix for CN states from likelihood data,
    # and computes CN probabilities for both autosomal and gonosomal samples.
    # The function adds an 'init' state to the transition matrix, improving its use
    # in Hidden Markov Models (HMMs).
    # The 'init' state, based on priors, helps to start and reset the HMM but doesn't
    # represent any actual CN state.
    # CNProbs_A and CNProbs_G dictionaries contain CN probabilities for autosomal and
    # gonosomal exon and for each samples, respectively, and are important for
    # further analysis, such as calculating quality scores and assessing CNV impacts.
    # The resulting 'transMatrix' is a 2D numpy array. dim =(nbOfCNStates + 1) * (nbOfCNStates + 1)
    try:
        (transMatrix, CNProbs_A, CNProbs_G) = callCNVs.transitions.getTransMatrixAndProbs(likelihoods_A, likelihoods_G,
                                                                                          autosomeExons, gonosomeExons,
                                                                                          priors, len(CNStates))
    except Exception as e:
        logger.error("getTransMatrix failed : %s", repr(e))
        raise Exception("getTransMatrix failed")

    thisTime = time.time()
    logger.debug("Done getTransMatrix, in %.2fs", thisTime - startTime)
    startTime = thisTime

    sys.exit()

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
