###############################################################################################
######################################## MAGE-CNV step 2: Sample clustering  ##################
###############################################################################################
# Given a TSV of fragment counts as produced by 1_countFrags.py:
# normalize the counts (in FPM = fragments per million), quality-control the samples,
# and build clusters of "comparable" samples that will be used as controls for one another.
# See usage for details.
###############################################################################################
import sys
import getopt
import os
import time
import logging
import numpy as np

####### MAGE-CNV modules
import countFrags.countsFile
import countFrags.countFragments
import clusterSamps.qualityControl
import clusterSamps.getGonosomesExonsIndexes
import clusterSamps.clustering
import clusterSamps.clustFile


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
    plotDir = "./ResultPlots/"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts, normalize the counts (Fragments Per Million), perform
quality controls on the samples and form the reference clusters for the call.
The execution of the default command separates autosomes ("A") and gonosomes ("G") for
clustering, to avoid bias (accepted sex chromosomes: X, Y, Z, W).
Results are printed to stdout in TSV format: 5 columns
[clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
In addition, all graphical support (quality control histogram for each sample and
dendogram from clustering) are printed in pdf files created in plotDir.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py
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
   --plotDir[str]: subdir (created if needed) where result plots files will be produced, default :  """ + plotDir + """
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

    # should density plots compare several different KDE bandwidth algorithms and values?
    # hard-coded here rather than set via parseArgs because this should only be set
    # to True for dev & testing
    testSmoothingBWs = False

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse counts
    try:
        (exons, samples, countsArray) = countFrags.countsFile.parseCountsFile(countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed for %s : %s", countsFile, repr(e))
        raise Exception("parseCountsFile failed")

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # normalize counts (FPM)
    try:
        countsFPM = countFrags.countFragments.normalizeCounts(countsArray)
    except Exception as e:
        logger.error("normalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("normalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done normalizing counts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # plot exon FPM densities for all samples; use this to identify QC-failing samples,
    # and exons with decent coverage in at least one sample (other exons can be ignored)
    plotFilePass = plotDir + "/coverageProfile_PASS.pdf"
    plotFileFail = plotDir + "/coverageProfile_FAIL.pdf"
    try:
        (sampsQCfailed, capturedExons) = clusterSamps.qualityControl.SampsQC(countsFPM, samples, plotFilePass,
                                                                             plotFileFail, testBW=testSmoothingBWs)
    except Exception as e:
        logger.error("SampsQC failed for %s : %s", countsFile, repr(e))
        raise Exception("SampsQC failed")

    thisTime = time.time()
    logger.debug("Done samples quality control, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # Clustering:
    # goal: establish the most optimal clustering(s) from the data validated
    # (exons and samples) by the quality control.

    maskGonoExIndexes = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)

    # - exonsToKeep (list of list[str,int,int,str]): contains exons information from captured exons
    exonsToKeep = [exons[i] for i, m in enumerate(capturedExons) if m]

    #####
    # Get Autosomes Clusters
    logger.info("### Autosomes, sample clustering:")
    try:
        autosomesFPM = countsFPM[~maskGonoExIndexes]
        plotAutoDendogramm = os.path.join(plotDir, "Dendogram_" + str(autosomesFPM.shape[1]) + "Samps_autosomes.pdf")
        # applying hierarchical clustering for autosomes exons
        autosClusters = clusterSamps.clustering.clustersBuilds(autosomesFPM, samples, maxCorr, minCorr, minSamps, plotAutoDendogramm)

    except Exception as e:
        logger.error("clusterBuilds for autosomes failed : %s", repr(e))
        raise Exception("clusterBuilds for autosomes failed")

    thisTime = time.time()
    logger.debug("Done samples clustering for autosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime

    #####
    # Get Gonosomes Clusters
    logger.info("### Gonosomes, sample clustering:")
    gonoFailed = False
    try:
        gonosomesFPM = countsFPM[maskGonoExIndexes]
        # need to test if there are gonosomal exons
        # e.g target sequencing
        # if not present no clustering returns a message in the log
        if gonosomesFPM.shape[0] != 0:
            plotGonoDendogramm = os.path.join(plotDir, "Dendogram_" + str(gonosomesFPM.shape[1]) + "Samps_gonosomes.pdf")
            gonosClusters = clusterSamps.clustering.clustersBuilds(gonosomesFPM, samples, maxCorr, minCorr, minSamps, plotGonoDendogramm)
        else:
            logger.info("No gonosomic exons, clustering can be done.")
            gonoFailed = True
    except Exception as e:
        logger.error("clusterBuilds for gonosomes failed : %s", repr(e))
        raise Exception("clusterBuilds for gonosomes failed")
    
    thisTime = time.time()
    logger.debug("Done samples clustering for gonosomes : in %.2fs", thisTime - startTime)
    startTime = thisTime
    
    # # unlike the processing carried out on the sets of chromosomes and autosomes,
    # # it is necessary for the gonosomes to identify the sample genders.
    # # This makes it possible to identify the clusters made up of the two genders
    # # which can potentially lead to copy number calls errors.
    # # e.g. in humans compared males and females: it is possible to observe
    # # heterodel calls on the X chromosome in males and homodel calls on the Y
    # # chromosome in females.
    # logger.info("### Gonosomes, gender prediction:")
    # try:
    #     # prediction of two groups based on kmeans from count data,
    #     # assignment of genders to each group based on sex chromosome coverage ratios.
    #     # obtaining 2 outputs:
    #     # - kmeans (list[int]): groupID predicted by Kmeans ordered on SOIsIndex
    #     # - sexePred (list[str]): genderID (e.g ["M","F"]), the order
    #     # correspond to KMeans groupID (0=M, 1=F)
    #     (kmeans, sexePred) = clusterSamps.genderDiscrimination.genderAttribution(validCountsFPM, gonoIndex, genderInfo)
    # except Exception as e:
    #     logger.error("genderAttribution ie gender prediction from gonosomes failed : %s", repr(e))
    #     raise Exception("genderAttribution failed")

    # thisTime = time.time()
    # logger.debug("Done gender prediction from gonosomes : in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # logger.info("### standardisation of results and validation:")
    # try:
    #     # grouping clustering results between autosomes and gonosomes
    #     # standardisation and verification of clustering results:
    #     # - clustsResList (list of lists[str,str,str,int,str]): to be printed in STDOUT
    #     # contains the following information: clusterID, list of samples added to compose the cluster,
    #     # clusterIDs controlling this cluster, validity of the cluster according to its total number
    #     # (<20 =invalid), its cluster status ("A" for clusters from autosomes, "G" for gonosomes and adding gender
    #     # composition "M" only males, "F" only females, "B" both genders are present in the cluster)
    #     # beware if a target cluster of one gender has a control cluster with the another gender,
    #     # this is not indicated by "B".
    #     clustsResList = clusterSamps.clustering.STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps,
    #                                                             noGender, clust2SampsGono, trgt2CtrlsGono, kmeans, sexePred)
    # except Exception as e:
    #     logger.error("STDZandCheckRes ie standardisation of results and validation failed : %s", repr(e))
    #     raise Exception("STDZandCheckRes failed")

    # thisTime = time.time()
    # logger.debug("Done standardisation of results and validation in %.2fs", thisTime - startTime)
    # startTime = thisTime

    # #####################################################
    # # print results
    # ##################
    # clusterSamps.clustering.printClustersFile(clustsResList, outFile)

    # thisTime = time.time()
    # logger.debug("Done printing results, in %.2fs", thisTime - startTime)
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
