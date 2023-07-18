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

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts, form the reference clusters for the call.
The default command execution pertains to exons and involves the separation of autosomes ('A')
and gonosomes ('G') for clustering to prevent bias. The accepted sex chromosomes are X, Y, Z, and W.
Results are printed to stdout in TSV format: 5 columns
[CLUSTER_ID, SAMPLES, CONTROLLED_BY, VALIDITY, SPECIFICS]
In addition, all graphical support (quality control histogram for each sample and
dendrogram from clustering) are printed in pdf files created in plotDir.
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
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "out=", "minSamps=", "maxCorr=", "minCorr=", "plotDir="])
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
    return(countsFile, outFile, minSamps, maxCorr, minCorr, plotDir)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (countsFile, outFile, minSamps, maxCorr, minCorr, plotDir) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse and FPM-normalize the counts, distinguishing between exons and intergenic pseudo-exons
    try:
        (samples, exons, intergenics, exonsFPM, intergenicsFPM) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # ##########################################
    # # data quality control
    # COMMENTED OUT FOR NOW - RE-ENABLE/ADAPT IF NEEDED, REMOVE CODE OTHERWISE
    # reason 1: Ignoring too many samples is bad, we should be able to make some
    # calls even for samples with lower-quality data
    # reason 2: with the new intergenic pseudo-exons, it should be possible to fit
    # a distribution for uncaptured exons, even for samples with an ugly FPM density plot.
    # However, it will be necessary to study the calling results for patients with
    # suspicious coverage density profiles before deleting this part.
    # ###################
    # # plot exon FPM densities for all samples; use this to identify QC-failing samples,
    # # and exons with decent coverage in at least one sample (other exons can be ignored)
    # # should density plots compare several different KDE bandwidth algorithms and values?
    # # hard-coded here rather than set via parseArgs because this should only be set
    # # to True for dev & testing
    # testSmoothingBWs = False
    #
    # plotFilePass = plotDir + "/coverageProfile_PASS.pdf"
    # plotFileFail = plotDir + "/coverageProfile_FAIL.pdf"
    # try:
    #     (sampsQCfailed, capturedExons) = clusterSamps.qualityControl.SampsQC(countsFPM, samples, plotFilePass,
    #                                                                          plotFileFail, testBW=testSmoothingBWs)
    # except Exception as e:
    #     logger.error("SampsQC failed for %s : %s", countsFile, repr(e))
    #     raise Exception("SampsQC failed")
    #
    # thisTime = time.time()
    # logger.debug("Done samples quality control, in %.2fs", thisTime - startTime)
    # startTime = thisTime

    ###################
    # Clustering:
    # build groups of samples with similar count profiles, independantly for:
    # - autosomes (gender-agnostic)
    # - sex chromosomes for Females
    # - sex chromosomes for Males
    # NOTE: clustering on gonosomes must be done separately for each gender, because we
    # use the Pearson Correlation Coefficient to identify similar samples... A man and
    # a woman can have PCC==1 on the X chromosome if the FPM counts are just doubled in
    # the woman, so these samples will be clustered together, but later we will call CNVs
    # everywhere on chrX

    exonOnSexChr = clusterSamps.genderPrediction.exonOnSexChr(exons)
    sample2gender = clusterSamps.genderPrediction.assignGender(exonsFPM, exonOnSexChr)

    thisTime = time.time()
    logger.debug("Done assigning genders to samples, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # subarrays of counts
    autosomesFPM = exonsFPM[exonOnSexChr == 0]
    gonosomesFemalesFPM = exonsFPM[exonOnSexChr != 0][:, sample2gender == 1]
    gonosomesMalesFPM = exonsFPM[exonOnSexChr != 0][:, sample2gender == 2]

    # autosomes
    try:
        plotFile = os.path.join(plotDir, "clusters_autosomes.pdf")
        clusters = clusterSamps.clustering.buildClusters(autosomesFPM, samples, maxCorr, minCorr, minSamps, plotFile)
    except Exception as e:
        logger.error("buildClusters failed for autosomes: %s", repr(e))
        raise Exception("buildClusters failed")

    thisTime = time.time()
    logger.debug("Done samples clustering for autosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime

    # gonosomes, only if we have at least one gonosomal exon and one Female/Male sample
    if gonosomesFemalesFPM.shape[0] != 0:
        try:
            plotFile = os.path.join(plotDir, "clusters_gonosomesFemale.pdf")
            clusters.extend(clusterSamps.clustering.buildClusters(gonosomesFemalesFPM, samples[sample2gender == 1],
                                                                  maxCorr, minCorr, minSamps, plotFile))
        except Exception as e:
            logger.error("buildClusters failed for gonosomes for Females: %s", repr(e))
            raise Exception("buildClusters failed")

    thisTime = time.time()
    logger.debug("Done samples clustering for gonosomes - Females, in %.2fs", thisTime - startTime)
    startTime = thisTime

    if gonosomesMalesFPM.shape[0] != 0:
        try:
            plotFile = os.path.join(plotDir, "clusters_gonosomesMale.pdf")
            clusters.extend(clusterSamps.clustering.buildClusters(gonosomesMalesFPM, samples[sample2gender == 2],
                                                                  maxCorr, minCorr, minSamps, plotFile))
        except Exception as e:
            logger.error("buildClusters failed for gonosomes for Males: %s", repr(e))
            raise Exception("buildClusters failed")

    thisTime = time.time()
    logger.debug("Done samples clustering for gonosomes - Males, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # print clustering results
    try:
        clusterSamps.clustFile.printClustsFile(clusters, outFile)
    except Exception as e:
        logger.error("printing clusters failed : %s", repr(e))
        raise Exception("printClustsFile failed")

    thisTime = time.time()
    logger.debug("Done printing clusters, in %.2fs", thisTime - startTime)
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
