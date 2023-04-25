###############################################################################################
######################################## MAGE-CNV step 2: Sample clustering  ##################
###############################################################################################
# Given a TSV of fragment counts as produced by 1_countFrags.py:
# build samples clusters that will be used as controls for one another.
# See usage for details.
###############################################################################################
import sys
import getopt
import os
import time
import logging

####### MAGE-CNV modules
import countFrags.countsFile
import clusterSamps.getGonosomesExonsIndexes
import clusterSamps.clustering
import clusterSamps.clustFile
import clusterSamps.genderPrediction

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


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
    outFile = ""
    # optional args with default values
    minSamps = 20
    maxCorr = 0.95
    minCorr = 0.85
    plotDir = "./plotDir/"
    sexPred = False

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts, form the reference clusters for the call.
The default command execution pertains to exons and involves the separation of autosomes ('A')
and gonosomes ('G') for clustering to prevent bias. The accepted sex chromosomes are X, Y, Z, and W.
Results are printed to stdout in TSV format: 5 columns
[CLUSTER_ID, SAMPLES, CONTROLLED_BY, VALIDITY, SPECIFICS]
In addition, all graphical support (quality control histogram for each sample and
dendogram from clustering) are printed in pdf files created in plotDir.
Optionally a prediction of the sexes per sample can be made empirically.

ARGUMENTS:
   --counts [str]: TSV file of fragment counts, possibly gzipped, produced by s1_countFrags.py
   --out [str] : file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
   --minSamps [int]: minimum number of samples to validate the creation of a cluster,
                     default : """ + str(minSamps) + """
   --maxCorr [float]: allows to define a Pearson correlation threshold according to which
                      the formation of clusters can start. Beware that a too small threshold
                      will lead to the formation of too small preliminary clusters, while a
                      too large threshold will lead to the formation of few but large clusters.
                      default: """ + str(maxCorr) + """
   --minCorr [float]: same principle as maxCorr but aims to complete the clustering.
                      Be careful, a low threshold will allow all the samples to be integrated
                      into clusters even if they are significantly different from the rest of
                      the clusters. A too high threshold will lead to a massive elimination of
                      non-clustered samples. default: """ + str(minCorr) + """
   --plotDir[str]: subdir (created if needed) where QC plot files will be produced, default:  """ + plotDir + """
   --sexPred (optional): if set, predict the sex of each sample, and append the predictions to the
                         outFile as two "clusters", one for each sex, listing the corresponding samples
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "out=", "minSamps=", "maxCorr=", "minCorr=", "plotDir=", "sexPred"])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
        elif opt in ("--out"):
            outFile = value
        elif (opt in ("--minSamps")):
            minSamps = value
        elif (opt in ("--maxCorr")):
            maxCorr = value
        elif (opt in ("--minCorr")):
            minCorr = value
        elif (opt in ("--plotDir")):
            plotDir = value
        elif (opt in ("--sexPred")):
            sexPred = True
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check mandatory args
    if countsFile == "":
        raise Exception("you must provide a countsFile with --counts. Try " + scriptName + " --help")
    elif not os.path.isfile(countsFile):
        raise Exception("countsFile " + countsFile + " doesn't exist")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other args
    try:
        minSamps = int(minSamps)
        if (minSamps <= 0):
            raise Exception()
    except Exception:
        raise Exception("minSamps must be a positive integer, not " + str(minSamps))

    try:
        maxCorr = float(maxCorr)
        if (maxCorr > 1) or (maxCorr < 0):
            raise Exception()
    except Exception:
        raise Exception("maxCorr must be a float between 0 and 1, not " + str(maxCorr))

    try:
        minCorr = float(minCorr)
        if (minCorr > maxCorr) or (minCorr < 0):
            raise Exception()
    except Exception:
        raise Exception("minCorr must be a float between 0 and maxCorr, not " + str(minCorr))

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, outFile, minSamps, maxCorr, minCorr, plotDir, sexPred)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (countsFile, outFile, minSamps, maxCorr, minCorr, plotDir, sexPred) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse counts, performs FPM normalization, distinguishes between intergenic regions and exons
    try:
        (samples, exons, intergenics, exonsFPM, intergenicsFPM) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # ##########################################
    # # data quality control suspension
    # # reason 1: Deleting too much patient sequencing data cannot be a safe option.
    # # reason 2: With the addition of intergenic windows it's possible to identify
    # # a profile of uncovered exons without a clean coverage densities per patient.
    # # However, it will be necessary to study the calling results for patients with
    # # dubious coverage profiles before deleting this part.
    # ###################
    # # plot exon FPM densities for all samples; use this to identify QC-failing samples,
    # # and exons with decent coverage in at least one sample (other exons can be ignored)
    # # should density plots compare several different KDE bandwidth algorithms and values?
    # # hard-coded here rather than set via parseArgs because this should only be set
    # # to True for dev & testing
    # testSmoothingBWs = False

    # plotFilePass = plotDir + "/coverageProfile_PASS.pdf"
    # plotFileFail = plotDir + "/coverageProfile_FAIL.pdf"
    # try:
    #     (sampsQCfailed, capturedExons) = clusterSamps.qualityControl.SampsQC(countsFPM, samples, plotFilePass,
    #                                                                          plotFileFail, testBW=testSmoothingBWs)
    # except Exception as e:
    #     logger.error("SampsQC failed for %s : %s", countsFile, repr(e))
    #     raise Exception("SampsQC failed")

    # thisTime = time.time()
    # logger.debug("Done samples quality control, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    ###################
    # Clustering:
    # goal: establish the most optimal clustering(s) from hierarchical clustering
    # need to segment the analysis between gonosomes and autosomes to avoid getting
    # aberrant CNs in the calling step.
    # (e.g Human : Male versus female => heterozygous CNV on the X)
    maskGonoExIndexes = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)

    #####
    # Get Autosomes Clusters
    logger.info("### Autosomes, sample clustering:")
    try:
        # get autosomes exons counts
        autosomesFPM = exonsFPM[~maskGonoExIndexes]
        plotAutoDendogramm = os.path.join(plotDir, "Dendogram_" + str(autosomesFPM.shape[1]) + "Samps_autosomes.pdf")
        # applying hierarchical clustering for autosomes exons
        autosClusters = clusterSamps.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps, plotAutoDendogramm)[0]

    except Exception as e:
        logger.error("clusterBuilds for autosomes failed : %s", repr(e))
        raise Exception("clusterBuilds for autosomes failed")

    thisTime = time.time()
    logger.debug("Done samples clustering for autosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime

    #####
    # Get Gonosomes Clusters
    logger.info("### Gonosomes, sample clustering:")
    try:
        gonosomesFPM = exonsFPM[maskGonoExIndexes]
        # Test if there are gonosomal exons
        # (e.g target sequencing may not be provided in some cases)
        # if not present, no clustering, returns a message in the log
        if gonosomesFPM.shape[0] != 0:
            plotGonoDendogramm = os.path.join(plotDir, "Dendogram_" + str(gonosomesFPM.shape[1]) + "Samps_gonosomes.pdf")
            gonosClusters = clusterSamps.clustering.clustersBuilds(gonosomesFPM, maxCorr, minCorr, minSamps, plotGonoDendogramm)[0]
        else:
            logger.info("no gonosomic exons, clustering can be done")
            gonosClusters = []
    except Exception as e:
        logger.error("clusterBuilds for gonosomes failed : %s", repr(e))
        raise Exception("clusterBuilds for gonosomes failed")

    thisTime = time.time()
    logger.debug("Done samples clustering for gonosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime

    ################
    # sex prediction
    sexAssign = []
    # case option is available, deduce the sex of the samples from the gonosomes coverage data.
    if sexPred:
        # checking for the presence of gonosomal exons
        # no gonosomal exons return an exception otherwise the assignment can be performed
        if gonosomesFPM.shape[0] != 0:
            try:
                gonosomesExons = [exons[i] for i in range(len(exons)) if maskGonoExIndexes[i]]
                sexAssign = clusterSamps.genderPrediction.sexAssignment(exonsFPM, gonosomesExons, samples)
            except Exception as e:
                logger.error("gender prediction failed: %s", repr(e))
                raise Exception("genderPrediction failed")

            thisTime = time.time()
            logger.debug("Done gender prediction, in %.2fs", thisTime - startTime)
            startTime = thisTime
        else:
            logger.error("no gonosomic exons, sex assignment can be done")

    ###################
    # print clustering results
    try:
        clusterSamps.clustFile.printClustsFile(autosClusters, gonosClusters, samples, sexAssign, outFile)
    except Exception as e:
        logger.error("printing results failed : %s", repr(e))
        raise Exception("printClustsFile failed")

    thisTime = time.time()
    logger.debug("Done printing results, in %.2fs", thisTime - startTime)
    startTime = thisTime

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
