###############################################################################################
######################## JACNEx step 3: exon filtering and calling ############################
###############################################################################################
# Given a TSV of exon fragment counts produced by 1_countFrags.py
# and a TSV with clustering information produced by 2_clusterSamps.py:
# It operates as the third step in the CNV analysis pipeline, emphasizing exon filtering and CNV calling.
# See usage for details.
###############################################################################################
import getopt
import logging
import numpy
import os
import sys
import time

####### JACNEx modules
import countFrags.countsFile
import countFrags.bed
import clusterSamps.clustFile
import callCNVs.priors
import callCNVs.likelihoodsNTM
import callCNVs.transitions
import callCNVs.callCNVs
import callCNVs.callsFile

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)
# override inherited level (when working on step 3)
logger.setLevel(logging.DEBUG)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
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
    madeBy = ""
    # optional args with default values
    padding = 10
    transMatrixCutoff = 90
    plotDir = "./plotDir/"
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Accepts exon fragment count data (from 1_countFrags.py) and sample clustering information
(from 2_clusterSamps.py) as input.
Performs several critical operations:
    a) Determines parameters for CN0 (half Gaussian) and CN2 (Gaussian) distributions for
       autosomal and gonosomal exons.
    b) Excludes non-interpretable exons based on set criteria.
    c) Calculates likelihoods for each CN state across exons and samples.
    d) Generates a transition matrix for CN state changes.
    e) Applies a Hidden Markov Model (HMM) to call and group CNVs.
    f) Outputs the CNV calls in VCF format.
The script utilizes multiprocessing for efficient computation and is structured to handle
errors and exceptions effectively, providing clear error messages for troubleshooting.
In addition, pie chart summarising exon filtering are produced as pdf files in plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 4 columns hold the sample cluster definitions.
                    [CLUSTER_ID, SAMPLES, FIT_WITH, VALID]. File obtained from 2_clusterSamps.py.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist.
    --madeBy [str]: program name + version to print as "source=" in the produced VCF.
    --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
    --transMatrixCutoff [int]: inter-exon percentile threshold for initializing the transition matrix,
                               default : """ + str(transMatrixCutoff) + """
    --plotDir [str]: subdir (created if needed) where plot files will be produced, default:  """ + plotDir + """
    --jobs [int]: cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "clusts=", "out=", "madeBy=",
                                                       "padding=", "transMatrixCutoff=", "plotDir=", "jobs="])
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
        elif opt in ("--madeBy"):
            madeBy = value
        elif opt in ("--padding"):
            padding = value
        elif opt in ("--transMatrixCutoff"):
            transMatrixCutoff = value
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

    if madeBy == "":
        raise Exception("you must provide a madeBy string with --madeBy. Try " + scriptName + " --help")

    #####################################################
    # Check other args
    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))

    try:
        transMatrixCutoff = int(transMatrixCutoff)
        if (transMatrixCutoff < 10):
            raise Exception()
        elif (transMatrixCutoff < 50):
            logger.warning("transMatrixCutoff smaller than 50 may be insufficient?")
    except Exception:
        raise Exception("transMatrixCutoff must be greater than 10, not " + str(transMatrixCutoff))

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    # AOK, return everything that's needed
    return (countsFile, clustsFile, outFile, padding, transMatrixCutoff, plotDir, jobs, madeBy)


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
    (countsFile, clustsFile, outFile, padding, pDist, plotDir, jobs, madeBy) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

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
    # parse clusters informations and returns a tuple containing four dictionaries:
    # - clust2samps: cluster to samples mapping,
    # - samp2clusts: sample to clusters mapping,
    # - fitWith: cluster to similar clusters mapping,
    # - clustIsValid: cluster validity status.
    try:
        (clust2samps, samp2clusts, fitWith, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parseClustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ###################
    # calculate metrics for building and adjusting the transition matrix.
    # These will be used later, but we calculate them now because they are
    # cluster-independant.
    allExons = autosomeExons.copy()
    allExons.extend(gonosomeExons)
    (baseTransMatMaxIED, adjustTransMatDMax) = countFrags.bed.getInterExonDistCutoffs(allExons)

    ####################
    # Exon profiling:
    # To assign specific CN profiles to each exon, applies continuous statistical distributions
    # to approximate CN profiles for each sample cluster and individual exon.

    # For the CN0 profile (homodeletion represented by uncaptured exon), the script uses data from
    # intergenic pseudo-exons.
    # Calculates key parameters of the best-fit half-Gaussian distribution for CN0 profiles
    # from 111 tested continuous distributions (scipy.stats.rv_continuous).
    # Determines a threshold (uncaptThreshold) to identify exons that are not captured
    # in the sequencing process, distinguishing them from captured exons.
    try:
        (hnorm_loc, hnorm_scale, uncaptThreshold) = callCNVs.exonProfiling.calcCN0Params(intergenicFPMs)
    except Exception as e:
        raise Exception("calcCN0Params failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done calcCN0Params, loc=%.2f, scale=%.2f, uncaptThreshold=%.2f in %.2f s",
                 hnorm_loc, hnorm_scale, uncaptThreshold, thisTime - startTime)
    startTime = thisTime

    # For the CN2 profile, performs robust fitting of a Gaussian distribution to determine
    # the CN2 profile, representing the normal or diploid state of exons.
    # Focuses on capturing the predominant signal, filtering out noise or less significant data.
    # Implements a distinct approach for autosomes and gonosomes to enhance accuracy.
    # Specific clusters and corresponding FPM data are used for each, accounting for the
    # unique characteristics of autosomal and gonosomal exons.
    # Calculates parameters for the Gaussian distribution representing CN2 profiles for each cluster.
    # The CN1 (single copy loss) and CN3+ (copy gain) profiles are deduced from the parameters
    # of the Gaussian distribution established for CN2. Based on the assumption that CN1 and
    # CN3 represent deviations from the CN2 state.
    try:
        (CN2Params_A, CN2Params_G) = callCNVs.exonProfiling.calcCN2Params(autosomeFPMs, gonosomeFPMs, samples,
                                                                          uncaptThreshold, clust2samps, fitWith,
                                                                          clustIsValid, plotDir, jobs)
    except Exception as e:
        raise Exception("calcCN2Params failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done calcCN2Params in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # Rescaling:
    # Rescaling FPMs ensure consistency across samples.
    # FPM values can vary due to factors like sequencing depth, necessitating rescaling for comparability.
    # Additionally, rescaling is crucial for adjusting parameters of probability distributions,
    # like the standard deviation of the half-normal distribution and standard deviation of the main normal distribution.
    # This adjustment ensures that statistical assumptions remain valid and models accurately reflect
    # data characteristics, enhancing the accuracy of subsequent analyses.
    # dictionaries params_A and params_G follow the same format as CN2Params but contain different data
    # [stdCN2/meanCN2, stdCN0/meanCN2].
    # The arrays FPMsrescal_A and FPMsrescal_G have the same dimensions as the original arrays autosomeFPMs and gonosomeFPMs,
    # respectively.
    try:
        (params_A, params_G, FPMsrescal_A, FPMsrescal_G) = callCNVs.rescaling.rescaleClusterFPMAndParams(autosomeFPMs, gonosomeFPMs,
                                                                                                         CN2Params_A, CN2Params_G,
                                                                                                         hnorm_scale, samples,
                                                                                                         clust2samps, fitWith,
                                                                                                         clustIsValid)
    except Exception as e:
        raise Exception("rescaleClusterFPMAndParams failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done rescaleClusterFPMAndParams in %.2f s", thisTime - startTime)
    startTime = thisTime

    ########################
    # Build HMM input datas:
    # determines copy states (e.g., diploid, deletion, duplication) for each genome region and sample
    # based on probabilistic data and state transitions, enabling the detection of significant genomic variations.

    # - States to be emitted by the HMM corresponding to different types of copy numbers.
    CNStates = ["CN0", "CN1", "CN2", "CN3"]

    ####################
    # - Calculates the likelihoods for each sample in a genomic dataset,
    # considering both autosomal and gonosomal data. It uses FPM data.
    # The function applies continuous distribution parameters (specifically designed to
    # describe the CN profile for each exon) to compute the likelihoods. These likelihoods
    # are essentially Pseudo Emission Probabilities, at the exon level in different samples.
    # The calculation is performed in parallel for efficiency, handling autosomes and gonosomes separately.
    try:
        (likelihoods_A, likelihoods_G) = callCNVs.likelihoods.calcLikelihoods(samples, FPMsrescal_A, FPMsrescal_G,
                                                                              clust2samps, clustIsValid, params_A,
                                                                              params_G, CNStates, jobs)
    except Exception as e:
        raise Exception("calcLikelihoods failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done calcLikelihoods in %.2f s", thisTime - startTime)
    startTime = thisTime

    # temp trick for comparing with AS implem: merge auto and gono likelihoodDicts, stacking the
    # gonosome likelihoods below thed autosome likelihoods for each sample
    likelihoodsDict = {}
    for s in likelihoods_A.keys():
        if s in likelihoods_G:
            likelihoodsDict[s] = numpy.vstack((likelihoods_A[s], likelihoods_G[s]))
        else:
            likelihoodsDict[s] = numpy.vstack((likelihoods_A[s],
                                               numpy.full((len(gonosomeExons), len(CNStates)), -1)))

    #########
    # Obtain priors probabilities using likelihood data for each CN.
    # The process follows Bayesian principles, which involve updating prior probabilities based on observed data.
    # This ensures that the estimation of prior probabilities aligns with the evidence provided by likelihood
    # data for each CN.
    # Bayesian theory provides a robust framework for this adjustment, facilitating convergence between
    # previous and current priors.
    try:
        priors = callCNVs.priors.calcPriors(likelihoodsDict, jobs)
    except Exception as e:
        raise Exception("calcPriors failed: %s", repr(e))

    formatted_priors = "  ".join(["%.2e" % x for x in priors])
    logger.info("Calculated priors: %s", formatted_priors)
    thisTime = time.time()
    logger.debug("Done calcPriors in %.2f s", thisTime - startTime)
    startTime = thisTime

    #########
    # - generates a transition matrix for CN states from likelihood data,
    # and computes CN probabilities for both autosomal and gonosomal samples.
    # The function adds an 'init' state to the transition matrix, improving its use
    # in Hidden Markov Models (HMMs).
    # The resulting 'transMatrix' is a 2D numpy array. dim =(nbOfCNStates) * (nbOfCNStates)
    try:
        transMatrix = callCNVs.transitions.buildBaseTransMatrix(likelihoodsDict, allExons,
                                                                priors, baseTransMatMaxIED)
    except Exception as e:
        raise Exception("buildBaseTransMatrix failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done buildBaseTransMatrix, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ######## DEBUG PRINT
    formatted_matrix = "\n".join(["\t".join([f"{cell:.2e}" for cell in row]) for row in transMatrix])
    logger.debug("Transition Matrix:\n%s", formatted_matrix)
    ########

    #########
    # Application of the HMM using the Viterbi algorithm. (calling step)
    # processes both autosomal and gonosomal exon data for a set of samples, yielding
    # a list of CNVs for each category. Each CNV is detailed with information including
    # CNV type, start and end positions of the affected exons, the event quality score,
    # and the sample ID.
    try:
        CNVs = callCNVs.callCNVs.callAllCNVs(likelihoodsDict, allExons, transMatrix,
                                             priors, adjustTransMatDMax, jobs)
    except Exception as e:
        raise Exception("callAllCNVs() failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done callAllCNVs() in %.2f s", thisTime - startTime)
    startTime = thisTime

    #########
    # VCF printing
    try:
        callCNVs.callsFile.printCallsFile(CNVs, allExons, samples, padding, outFile, madeBy)
    except Exception as e:
        raise Exception("printCallsFile failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done printCallsFile in %.2f s", thisTime - startTime)
    startTime = thisTime


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
