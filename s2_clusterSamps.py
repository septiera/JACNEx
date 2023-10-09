###############################################################################################
######################################## MAGE-CNV step 2: Sample clustering  ##################
###############################################################################################
# Given a TSV of fragment counts as produced by 1_countFrags.py:
# build clusters of samples that will be used as controls for one another.
# See usage for details.
###############################################################################################
import getopt
import logging
import os
import sys
import time
import traceback

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
    plotDir = "./plotDir/"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts, build clusters of "comparable" samples that
will be used as controls for one another.
Clusters are built independantly for exons on autosomes ('A') and  on gonosomes ('G').
The accepted sex chromosomes are X, Y, Z, and W.
Results are printed to --out in TSV format: 4 columns
[CLUSTER_ID, SAMPLES, FIT_WITH, VALID]
In addition, dendrograms of the clustering results are produced as pdf files in plotDir.

ARGUMENTS:
   --counts [str]: TSV file of fragment counts, possibly gzipped, produced by s1_countFrags.py
   --out [str] : file where clusters will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
   --minSamps [int]: minimum number of samples for a cluster to be declared valid, default : """ + str(minSamps) + """
   --plotDir [str]: subdir (created if needed) where plot files will be produced, default:  """ + plotDir + """
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "out=", "minSamps=", "plotDir="])
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

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, outFile, minSamps, plotDir)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (countsFile, outFile, minSamps, plotDir) = parseArgs(argv)

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
    logger.info("done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
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
    # build clusters of samples with "similar" count profiles, independently for exons
    # located on autosomes and on sex chromosomes (==gonosomes).
    # As a side benefit this also allows to identify same-gender samples, since samples
    # that cluster together for the gonosomal exons always have the same gonosomal
    # karyotype (predominantly on the X: in our hands XXY samples always cluster with XX
    # samples, not with XY ones)

    # subarrays of counts
    exonOnSexChr = clusterSamps.genderPrediction.exonOnSexChr(exons)
    autosomesFPM = exonsFPM[exonOnSexChr == 0]
    gonosomesFPM = exonsFPM[exonOnSexChr != 0]

    # build root name for dendrograms, will just need to append autosomes.pdf or gonosomes.pdf
    dendroFileRoot = os.path.basename(outFile)
    # remove file extension (.tsv probably), and also .gz if present
    if dendroFileRoot.endswith(".gz"):
        dendroFileRoot = os.path.splitext(dendroFileRoot)[0]
    dendroFileRoot = os.path.splitext(dendroFileRoot)[0]
    dendroFileRoot = "dendrogram_" + dendroFileRoot
    dendroFileRoot = os.path.join(plotDir, dendroFileRoot)

    # autosomes
    try:
        plotFile = dendroFileRoot + "_autosomes.pdf"
        (clust2samps, fitWith, clustIsValid) = clusterSamps.clustering.buildClusters(
            autosomesFPM, "A", samples, minSamps, plotFile)
    except Exception as e:
        logger.error("buildClusters failed for autosomes: %s", repr(e))
        traceback.print_exc()
        raise Exception("buildClusters failed")

    thisTime = time.time()
    logger.info("done clustering samples for autosomes, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # sex chromosomes
    try:
        plotFile = dendroFileRoot + "_gonosomes.pdf"
        (clust2sampsGono, fitWithGono, clustIsValidGono) = clusterSamps.clustering.buildClusters(
            gonosomesFPM, "G", samples, minSamps, plotFile)
    except Exception as e:
        logger.error("buildClusters failed for gonosomes: %s", repr(e))
        traceback.print_exc()
        raise Exception("buildClusters failed")
    clust2samps.update(clust2sampsGono)
    fitWith.update(fitWithGono)
    clustIsValid.update(clustIsValidGono)

    thisTime = time.time()
    logger.info("done clustering samples for gonosomes, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # print clustering results
    try:
        clusterSamps.clustFile.printClustsFile(clust2samps, fitWith, clustIsValid, outFile)
    except Exception as e:
        logger.error("printing clusters failed : %s", repr(e))
        raise Exception("printClustsFile failed")

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
        sys.stderr.write("ERROR in " + scriptName + " : " + repr(e) + "\n")
        traceback.print_exc()
        sys.exit(1)
