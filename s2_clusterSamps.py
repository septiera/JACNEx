###############################################################################################
######################################## MAGE-CNV step 2: Sample clustering  ##################
###############################################################################################
# Given a TSV of fragment counts as produced by 1_countFrags.py:
# normalize the counts (in FPM = fragments per million), quality-control the samples,
# and build clusters of "comparable" samples that will be used as controls for one another.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import time
import logging
import numpy as np

####### MAGE-CNV modules
import countFrags.countsFile
import clusterSamps.normalisation
import clusterSamps.qualityControl
import clusterSamps.genderDiscrimination
import clusterSamps.clustering


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])
    # mandatory args
    countsFile = ""
    # optionnal args with default values
    minSamps = "20"
    maxCorr = "0.95"
    minCorr = "0.85"
    plotDir = "./ResultPlots/"
    # boolean args with False status by default
    nogender = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million), performs
quality control on the samples and forms the reference clusters for the call.
The execution of the default command separates autosomes ("A") and gonosomes ("G") for
clustering, to avoid bias (accepted sex chromosomes: X, Y, Z, W).
Results are printed to stdout in TSV format: 5 columns
[clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
In addition, all graphical support (quality control histogram for each sample and
dendogram from clustering) are printed in pdf files created in plotDir.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py

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
   --plotDir[str]: subdir (created if needed) where result plots files will be produced, default :  """ + str(plotDir) + """
   --nogender [boolean]: no gender discrimination for clustering. Calling the argument is sufficient.
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "out=", "minSamps=", "maxCorr=", "minCorr=", "plotDir=", "nogender"])
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : " + e.msg + ". Try " + scriptName + " --help\n")
        raise Exception()

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            raise Exception()
        elif (opt in ("--counts")):
            countsFile = value
        elif (opt in ("--minSamps")):
            minSamps = value
        elif (opt in ("--maxCorr")):
            maxCorr = value
        elif (opt in ("--minCorr")):
            minCorr = value
        elif (opt in ("--plotDir")):
            plotDir = value
        elif (opt in ("--nogender")):
            nogender = True

        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()

    #####################################################
    # Check that the mandatory parameters
    if countsFile == "":
        sys.stderr.write("ERROR : You must provide a TSV file with counts with --counts. Try " + scriptName + " --help.\n")
        raise Exception()
    elif (not os.path.isfile(countsFile)):
        sys.stderr.write("ERROR : countsFile " + countsFile + " doesn't exist.\n")
        raise Exception()

    #####################################################
    # Check other args
    if minSamps == "":
        sys.stderr.write("ERROR : You must provide an integer value with --minSamps. Try " + scriptName + " --help.\n")
        raise Exception()
    else:
        try:
            minSamps = np.int(minSamps)
            if (minSamps < 0):
                raise Exception()
        except Exception:
            sys.stderr.write("ERROR : minSamps must be a non-negative integer, not '" + minSamps + "'.\n")
            raise Exception()

    if maxCorr == "":
        sys.stderr.write("ERROR : You must provide a float value with --maxCorr. Try " + scriptName + " --help.\n")
        raise Exception()
    else:
        try:
            maxCorr = np.float(maxCorr)
            if (maxCorr > 1 or maxCorr < 0):
                raise Exception()
        except Exception:
            sys.stderr.write("ERROR : maxCorr must be a float between 0 and 1, not '" + maxCorr + "'.\n")
            raise Exception()

    if minCorr == "":
        sys.stderr.write("ERROR : You must provide a float value with --minCorr. Try " + scriptName + " --help.\n")
        raise Exception()
    else:
        try:
            minCorr = np.float(minCorr)
            if (minCorr > 1 or minCorr < 0):
                raise Exception()
        except Exception:
            sys.stderr.write("ERROR : minCorr must be a float between 0 and 1, not '" + minCorr + "'.\n")
            raise Exception()

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception:
            sys.stderr.write("ERROR : plotDir " + plotDir + " doesn't exist and can't be mkdir'd\n")
            raise Exception()

    # AOK, return everything that's needed
    return(countsFile, minSamps, maxCorr, minCorr, plotDir, nogender)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (countsFile, minSamps, maxCorr, minCorr, plotDir, nogender) = parseArgs(argv)

    ################################################
    # args seem OK, start working
    logger.info("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    #######################################################
    # Parse TSV file of counts
    ###################
    # To obtain :
    # - exons (list of lists[str,int,int,str]): CHR,START,END,EXON_ID
    #   the exons are sorted according to their genomic position and padded.
    # - SOIs (list[str]): sampleIDs copied from countsFile's header
    # - countsArray (np.ndarray[int]): fragment counts, dim = NbExons x NbSOIs
    try:
        (exons, SOIs, countsArray) = countFrags.countsFile.parseCountsFile(countsFile)
    except Exception:
        logger.error("parseCountsFile failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Normalisation:
    ##################
    # Fragment counts are standardised in Fragment Per Million (FPM).
    # - FPMArray (np.ndarray[float]): normalised counts of countsArray same dimension
    #   for arrays in input/output: NbExons*NbSOIs
    try:
        FPMArray = clusterSamps.normalisation.FPMNormalisation(countsArray)
    except Exception:
        logger.error("FPMNormalisation failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done fragment counts normalisation, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Quality control:
    ##################
    # sample coverage profil validity assessment and identification of exons with
    # little or no coverage common to the validated samples
    # - sampsQCfailed (list[int]): sample indexes not validated by quality control
    # - uncoveredExons (list[int]): exons indexes with little or no coverage common
    #   to all samples passing quality control
    try:
        QCPDF = os.path.join(plotDir, "CoverageProfilChecking_" + str(len(SOIs)) + "samps.pdf")
        (sampsQCfailed, uncoveredExons) = clusterSamps.qualityControl.SampsQC(FPMArray, SOIs, QCPDF)

    except Exception as e:
        logger.error("SampQC failed %s", e)
        raise Exception()

    thisTime = time.time()
    logger.debug("Done samples quality control, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Clustering:
    ####################
    # objective: establish the most optimal clustering(s) from the data validated
    # (exons and samples) by the quality control.

    ####################
    # filtering the counts data to recover valid samples and covered exons
    # - validFPMArray (np.ndarray[float]): normalized fragment counts for exons covered
    # for all samples that passed quality control
    validFPMArray = np.delete(FPMArray, sampsQCfailed, axis=1)
    validFPMArray = np.delete(validFPMArray, uncoveredExons, axis=0)

    ###########################
    #### no gender discrimination
    # clustering algorithm direct application
    if nogender:
        logger.info("### Samples clustering:")
        try:
            dendogramPDF = os.path.join(plotDir, "Dendogram_" + str(len(SOIs)) + "Samps_FullChrom.pdf")
            # applying hierarchical clustering and obtaining 2 outputs:
            # - clust2Samps (dict(int : list[int])): clusterID associated to SOIsIndex
            #   key = clusterID, value = list of SOIsIndex
            # - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
            #   key = target clusterID, value = list of controls clusterID
            (clust2Samps, trgt2Ctrls) = clusterSamps.clustering.clustersBuilds(validFPMArray, maxCorr, minCorr, minSamps, dendogramPDF)
        except Exception:
            logger.error("clusterBuilding failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done clusterisation in %.2f s", thisTime - startTime)
        startTime = thisTime

        logger.info("### standardisation of results and validation:")
        try:
            # standardisation and verification of clustering results:
            # - clustsResList (list of lists[str,str,str,int,str]): to be printed in STDOUT
            # contains the following information: clusterID, list of samples added to compose the cluster,
            # clusterIDs controlling this cluster, validity of the cluster according to its total number
            # (<20 =invalid), its cluster status (here as all chromosomes are analysed together = "W" for whole)
            clustsResList = clusterSamps.clustering.STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps, nogender)

        except Exception:
            logger.error("standardisation of results and validation failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done standardisation of results and validation in %.2f s", thisTime - startTime)
        startTime = thisTime

    ###########################
    #### gender discrimination
    else:

        # - exonsToKeep (list of list[str,int,int,str]): contains exons information from covered exons
        exonsToKeep = [val for i, val in enumerate(exons) if i not in uncoveredExons]

        try:
            # parse exons to extract information related to the organisms studied and their gender
            # - gonoIndex (dict(str: list(int))): is a dictionary where key=GonosomeID(e.g 'chrX'),
            # value=list of gonosome exon index.
            # - genderInfo (list of list[str]):contains informations for the gender
            # identification, ie ["gender identifier","specific chromosome"].
            (gonoIndex, genderInfo) = clusterSamps.genderDiscrimination.getGenderInfos(exonsToKeep)
        except Exception:
            logger.error("getGenderInfos failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime - startTime)
        startTime = thisTime

        # cutting normalized count data
        # - gonoIndexFlat (np.ndarray[int]): flat gonosome exon indexes list
        gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist])
        # - autosomesFPM (np.ndarray[float]): normalized fragment counts for valid samples, exons covered
        # in autosomes
        autosomesFPM = np.delete(validFPMArray, gonoIndexFlat, axis=0)
        # - gonosomesFPM (np.ndarray[float]): normalized fragment counts for valid samples, exons covered
        # in gonosomes
        gonosomesFPM = np.take(validFPMArray, gonoIndexFlat, axis=0)

        #####################################################
        # Get Autosomes Clusters
        ##################
        logger.info("### Autosomes, sample clustering:")
        try:
            dendogramPDF = os.path.join(plotDir, "Dendogram_" + str(autosomesFPM.shape[1]) + "Samps_autosomes.pdf")
            # applying hierarchical clustering and obtaining 2 outputs autosome-specific:
            # - clust2Samps (dict(int : list[int])): clusterID associated to SOIsIndex
            #   key = clusterID, value = list of SOIsIndex
            # - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
            #   key = target clusterID, value = list of controls clusterID
            (clust2Samps, trgt2Ctrls) = clusterSamps.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps, dendogramPDF)

        except Exception:
            logger.error("clusterBuilds for autosomes failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done samples clustering for autosomes : in %.2f s", thisTime - startTime)
        startTime = thisTime

        #####################################################
        # Get Gonosomes Clusters
        ##################
        logger.info("### Gonosomes, sample clustering:")
        try:
            dendogramPDF = os.path.join(plotDir, "Dendogram_" + str(gonosomesFPM.shape[1]) + "Samps_gonosomes.png")
            # applying hierarchical clustering and obtaining 2 outputs gonosome-specific:
            # - clust2SampsGono (dict(int : list[int])): clusterID associated to SOIsIndex
            #   key = clusterID, value = list of SOIsIndex
            # - trgt2CtrlsGono (dict(int : list[int])): target and controls clusters correspondance,
            #   key = target clusterID, value = list of controls clusterID
            (clust2SampsGono, trgt2CtrlsGono) = clusterSamps.clustering.clustersBuilds(gonosomesFPM, maxCorr, minCorr, minSamps, dendogramPDF)

        except Exception:
            logger.error("clusterBuilds for gonosomesFPM failed")
            raise Exception()

        # unlike the processing carried out on the sets of chromosomes and autosomes,
        # it is necessary for the gonosomes to identify the sample genders.
        # This makes it possible to identify the clusters made up of the two genders
        # which can potentially lead to copy number calls errors.
        # e.g. in humans compared males and females: it is possible to observe
        # heterodel calls on the X chromosome in males and homodel calls on the Y
        # chromosome in females.
        logger.info("### Gonosomes, gender prediction:")
        try:
            # prediction of two groups based on kmeans from count data,
            # assignment of genders to each group based on sex chromosome coverage ratios.
            # obtaining 2 outputs:
            # - kmeans (list[int]): groupID predicted by Kmeans ordered on SOIsIndex
            # - sexePred (list[str]): genderID (e.g ["M","F"]), the order
            # correspond to KMeans groupID (0=M, 1=F)
            (kmeans, sexePred) = clusterSamps.genderDiscrimination.genderAttribution(validFPMArray, gonoIndex, genderInfo)
        except Exception:
            logger.error("gender prediction from gonosomes failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done gender prediction from gonosomes : in %.2f s", thisTime - startTime)
        startTime = thisTime

        logger.info("### standardisation of results and validation:")
        try:
            # grouping clustering results between autosomes and gonosomes
            # standardisation and verification of clustering results:
            # - clustsResList (list of lists[str,str,str,int,str]): to be printed in STDOUT
            # contains the following information: clusterID, list of samples added to compose the cluster,
            # clusterIDs controlling this cluster, validity of the cluster according to its total number
            # (<20 =invalid), its cluster status ("A" for clusters from autosomes, "G" for gonosomes and adding gender
            # composition "M" only males, "F" only females, "B" both genders are present in the cluster)
            # beware if a target cluster of one gender has a control cluster with the another gender,
            # this is not indicated by "B".
            clustsResList = clusterSamps.clustering.STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps,
                                                               nogender, clust2SampsGono, trgt2CtrlsGono, kmeans, sexePred)
        except Exception:
            logger.error("standardisation of results and validation failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done standardisation of results and validation in %.2f s", thisTime - startTime)
        startTime = thisTime

    #####################################################
    # print results
    ##################
    clusterSamps.clustering.printClustersFile(clustsResList)

    thisTime = time.time()
    logger.debug("Done printing results, in %.2f s", thisTime - startTime)
    logger.info("ALL DONE")


####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    try:
        main(sys.argv)
    except Exception:
        # whoever raised the exception should have explained it on stderr, here we just die
        exit(1)
