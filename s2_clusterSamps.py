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
Given a TSV of exon fragment counts, build clusters of "comparable" samples that
will be used as controls for one another.
Clusters are built independantly for exons on autosomes ('A') and  on gonosomes ('G').
The accepted sex chromosomes are X, Y, Z, and W.
Results are printed to --out in TSV format: 4 columns
[CLUSTER_ID, SAMPLES, FIT_WITH, VALID]
In addition, all graphical support (quality control histogram for each sample and
dendrogram from clustering) are produced as pdf files in plotDir.

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
    # build clusters of samples with "similar" count profiles, independantly for exons
    # located on autosomes and on sex chromosomes (==gonosomes).
    # As a side benefit this also allows to identify same-gender samples, since samples
    # that cluster together for the gonosomal exons always have the same gonosomal
    # karyotype (predominantly on the X: in our hands XXY samples always cluster with XX
    # samples, not with XY ones)

    # subarrays of counts
    exonOnSexChr = clusterSamps.genderPrediction.exonOnSexChr(exons)
    autosomesFPM = exonsFPM[exonOnSexChr == 0]
    gonosomesFPM = exonsFPM[exonOnSexChr != 0]

    # NOTE: in some tests, buildClusters() normalizes each sample in the PCA space before
    # the hierarchical clustering. Goal was to be able to use the same startDist/maxDist
    # for autosomes and gonosomes, because currently the distances are very differently
    # scaled... Results look good but I am using 500-1500 for autosomes, vs 100-300 for
    # gonosomes.
    # In the normalized approach, startDist and maxDist are the same for autosomes
    # and gonosomes, and should be robust
    normalize = True
    # defaults for startDist and maxDist, if normalize==False they will be changed
    # before calling buildClusters()
    startDist = 0.4
    maxDist = 1.0

    # autosomes
    try:
        if not normalize:
            startDist = 500
            maxDist = 1500
        plotFile = os.path.join(plotDir, "clusters_autosomes.pdf")
        (clust2samps, fitWith, clustIsValid, linkageMatrix) = clusterSamps.clustering.buildClusters(
            autosomesFPM, "A", samples, startDist, maxDist, minSamps, plotFile, normalize)
    except Exception as e:
        logger.error("buildClusters failed for autosomes: %s", repr(e))
        raise Exception("buildClusters failed")

    thisTime = time.time()
    logger.debug("Done clustering samples for autosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime

    # sex chromosomes
    try:
        if not normalize:
            startDist = 100
            maxDist = 300
        plotFile = os.path.join(plotDir, "clusters_gonosomes.pdf")
        (clust2sampsSex, fitWithSex, clustIsValidSex, linkageMatrixSex) = clusterSamps.clustering.buildClusters(
            gonosomesFPM, "G", samples, startDist, maxDist, minSamps, plotFile, normalize)
    except Exception as e:
        logger.error("buildClusters failed for gonosomes: %s", repr(e))
        raise Exception("buildClusters failed")
    clust2samps.update(clust2sampsSex)
    fitWith.update(fitWithSex)
    clustIsValid.update(clustIsValidSex)

    thisTime = time.time()
    logger.debug("Done clustering samples for gonosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # print clustering results
    try:
        clusterSamps.clustFile.printClustsFile(clust2samps, fitWith, clustIsValid, outFile)
    except Exception as e:
        logger.error("printing clusters failed : %s", repr(e))
        raise Exception("printClustsFile failed")

    thisTime = time.time()
    logger.debug("Done printing clusters, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # Code for studying gender predictions, based on sums of FPMs for exons on X or Y.
    # Based on these tests, gender prediction should be easy since we could clearly
    # distinguish the karyotypes on simple 1D plots... For example XXY samples were
    # in Female groups on the X and in Male groups on the Y. Also we have a single
    # patient with an XYY karyotype, and even that was quite apparent.
    # However this is no longer necessary, clustering now works for gonosomes thanks
    # to the PCC -> euclidean move.
    # sample2gender = clusterSamps.genderPrediction.assignGender(exonsFPM, exonOnSexChr, intergenicsFPM, samples)
    # gonosomesFemalesFPM = exonsFPM[exonOnSexChr != 0][:, sample2gender == 1]
    # gonosomesMalesFPM = exonsFPM[exonOnSexChr != 0][:, sample2gender == 2]
    # thisTime = time.time()
    # logger.debug("Done assigning genders to samples, in %.2fs", thisTime - startTime)
    # startTime = thisTime

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
