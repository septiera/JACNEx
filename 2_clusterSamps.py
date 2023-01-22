###############################################################################################
######################################## MAGE-CNV step 2: Sample clustering  ##################
###############################################################################################
# Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million), performs
# quality control on the samples and forms the reference clusters for the call.
# Prints results in a folder defined by the user.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import numpy as np
import time
import logging

####### MAGE-CNV modules
import mageCNV.countsFile
import mageCNV.normalisation
import mageCNV.qualityControl
import mageCNV.genderDiscrimination
import mageCNV.clustering


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
    outFolder = ""
    # optionnal args with default values
    minSamps = "20"
    maxCorr = "0.95"
    minCorr = "0.85"
    # boolean args with False status by default
    nogender = False
    figure = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million), performs
quality control on the samples and forms the reference clusters for the call.
The execution of the default command separates autosomes ("A") and gonosomes ("G") for
clustering, to avoid bias (accepted sex chromosomes: X, Y, Z, W).
Produces a single TSV file listing the clustering results.
By default no result figure is obtained, otherwise a pdf illustrating the data used for QC
is generated as well as one or more png's with the clustering dendogram(s) obtained.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py
   --out[str]: acces path to a pre-existing folder to save the output files
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
   --nogender [boolean]: no gender discrimination for clustering. Calling the argument is sufficient.
   --figure [boolean]: make histogramms and dendogram(s) that will be present in the output in
                       pdf and png format. Calling the argument is sufficient.\n
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "out=", "minSamps=", "maxCorr=", "minCorr=", "nogender", "figure"])
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
        elif (opt in ("--out")):
            outFolder = value
        elif (opt in ("--minSamps")):
            minSamps = value
        elif (opt in ("--maxCorr")):
            maxCorr = value
        elif (opt in ("--minCorr")):
            minCorr = value
        elif (opt in ("--nogender")):
            nogender = True
        elif (opt in ("--figure")):
            figure = True
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

    if outFolder == "":
        sys.stderr.write("ERROR : You must provide a folder path with --out. Try " + scriptName + " --help.\n")
        raise Exception()
    elif (not os.path.isdir(outFolder)):
        sys.stderr.write("ERROR : outFolder " + outFolder + " doesn't exist.\n")
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
            maxCorr = np.float(value)
            if (maxCorr > 1 or maxCorr < 0):
                raise Exception()
        except Exception:
            sys.stderr.write("ERROR : maxCorr must be a float between 0 and 1, not '" + value + "'.\n")
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
            sys.stderr.write("ERROR : minCorr must be a float between 0 and 1, not '" + value + "'.\n")
            raise Exception()

    # AOK, return everything that's needed
    return(countsFile, outFolder, minSamps, maxCorr, minCorr, nogender, figure)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (countsFile, outFolder, minSamps, maxCorr, minCorr, nogender, figure) = parseArgs(argv)

    ################################################
    # args seem OK, start working
    logger.info("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    #######################################################
    # Parse TSV file of counts
    ###################
    # To obtain :
    # - exons (list of lists[str,int,int,str]): information on exon , containing CHR,START,END,EXON_ID
    #   the exons are sorted according to their genomic position and padded.
    # - SOIs (list[str]): sampleIDs copied from countsFile's header
    # - countsArray (np.ndarray[int]): fragment counts, dim = NbExons x NbSOIs
    try:
        (exons, SOIs, countsArray) = mageCNV.countsFile.parseCountsFile(countsFile)
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
        FPMArray = mageCNV.normalisation.FPMNormalisation(countsArray)
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
        if figure:
            outputFile = os.path.join(outFolder, "CoverageProfilChecking_" + str(len(SOIs)) + "samps.pdf")
            (sampsQCfailed, uncoveredExons) = mageCNV.qualityControl.SampsQC(FPMArray, SOIs, outputFile)
        else:
            (sampsQCfailed, uncoveredExons) = mageCNV.qualityControl.SampsQC(FPMArray, SOIs)
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
            if figure:
                outputFile = os.path.join(outFolder, "Dendogram_" + str(len(SOIs)) + "Samps_FullChrom.png")
                # applying hierarchical clustering and obtaining 2 outputs:
                # - clust2Samps (dict(int : list[int])): clusterID associated to SOIsIndex
                #   key = clusterID, value = list of SOIsIndex
                # - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
                #   key = target clusterID, value = list of controls clusterID
                (clust2Samps, trgt2Ctrls) = mageCNV.clustering.clustersBuilds(validFPMArray, maxCorr, minCorr, minSamps, outputFile)
            else:
                (clust2Samps, trgt2Ctrls) = mageCNV.clustering.clustersBuilds(validFPMArray, maxCorr, minCorr, minSamps)

        except Exception:
            logger.error("clusterBuilding failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done clusterisation in %.2f s", thisTime - startTime)
        startTime = thisTime

        logger.info("### standardisation of results and validation:")

        try:
            # 
            clustsResList = mageCNV.clustering.STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps, nogender)

        except Exception:
            logger.error("standardisation of results and validation failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done standardisation of results and validation in %.2f s", thisTime - startTime)
        startTime = thisTime

    ###########################
    #### gender discrimination
    # avoid grouping Male with Female which leads to dubious CNV calls
    else:

        # - exonsToKeep (list of list[str,int,int,str]): contains information about the covered exons
        exonsToKeep = [val for i, val in enumerate(exons) if i not in uncoveredExons]

        try:
            # parse exons to extract information related to the organisms studied and their gender
            # - gonoIndex (dict(str: list(int))): is a dictionary where key=GonosomeID(e.g 'chrX'),
            # value=list of gonosome exon index.
            # - genderInfo (list of list[str]):contains informations for the gender
            # identification, ie ["gender identifier","specific chromosome"].
            (gonoIndex, genderInfo) = mageCNV.genderDiscrimination.getGenderInfos(exonsToKeep)
        except Exception:
            logger.error("getGenderInfos failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime - startTime)
        startTime = thisTime

        # cutting normalized count data according to autosomal exons
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
            if figure:
                outputFile = os.path.join(outFolder, "Dendogram_" + str(autosomesFPM.shape[1]) + "Samps_autosomes.png")
                (clust2Samps, trgt2Ctrls) = mageCNV.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps, outputFile)
            else:
                (clust2Samps, trgt2Ctrls) = mageCNV.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps)
        except Exception:
            logger.error("clusterBuilds for autosomes failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done samples clustering for autosomes : in %.2f s", thisTime - startTime)
        startTime = thisTime

        #####################################################
        # Get Gonosomes Clusters
        ##################
        # In contrast to all chromosomes and autosomes where only one clustering
        # analysis is expected, for gonosomes it is preferable to perform clustering
        # analyses by separating the genders.
        # This avoids, among other things, future copy number calling errors.
        # e.g. in humans compared males and females: it is possible to observe
        # heterodel calls on the X chromosome in males and homodel calls on the Y
        # chromosome in females.
        logger.info("### Gonosomes, gender prediction:")
        try:
            (kmeans, sexePred) = mageCNV.genderDiscrimination.genderAttribution(validFPMArray , gonoIndex, genderInfo)
        except Exception:
            logger.error("gender prediction from gonosomes failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done gender prediction from gonosomes : in %.2f s", thisTime - startTime)
        startTime = thisTime

        logger.info("### Gonosomes, sample clustering:")
        try:
            if figure:
                outputFile = os.path.join(outFolder, "Dendogram_" + str(gonosomesFPM.shape[1]) + "Samps_gonosomes.png")
                (clust2SampsGono, trgt2CtrlsGono) = mageCNV.clustering.clustersBuilds(gonosomesFPM, maxCorr, minCorr, minSamps, outputFile)
            else:
                (clust2SampsGono, trgt2CtrlsGono) = mageCNV.clustering.clustersBuilds(gonosomesFPM, maxCorr, minCorr, minSamps)
        except Exception:
            logger.error("clusterBuilds for gonosomesFPM failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done samples clustering for gonosomesFPM : in %.2f s", thisTime - startTime)
        startTime = thisTime

        logger.info("### standardisation of results and validation:")

        try:
            # 
            clustsResList = mageCNV.clustering.STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps, nogender,clust2SampsGono, trgt2CtrlsGono, kmeans, sexePred)

        except Exception:
            logger.error("standardisation of results and validation failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done standardisation of results and validation in %.2f s", thisTime - startTime)
        startTime = thisTime


    #####################################################
    # print results
    ##################
    mageCNV.clustering.printClustersFile(clustsResList)

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
