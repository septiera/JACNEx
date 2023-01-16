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
    minSamps = 20
    maxCorr = 0.95
    minCorr = 0.85
    # boolean args with False status by default
    nogender = False
    figure = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million), performs
quality control on the samples and forms the reference clusters for the call.
The execution of the default command, separates autosomes ("A") and gonosomes ("G") for
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
   --nogender[optionnal]: no gender discrimination for clustering
   --figure[optionnal]: make histogramms and dendogram(s) that will be present in the output in
                        pdf and png format\n"""

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
            if (not os.path.isfile(countsFile)):
                sys.stderr.write("ERROR : countsFile " + countsFile + " doesn't exist.\n")
                raise Exception()
        elif (opt in ("--out")):
            outFolder = value
            if (not os.path.isdir(outFolder)):
                sys.stderr.write("ERROR : outFolder " + outFolder + " doesn't exist.\n")
                raise Exception()
        elif (opt in ("--minSamps")):
            try:
                minSamps = np.int(value)
                if (minSamps < 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : minSamps must be a non-negative integer, not '" + value + "'.\n")
                raise Exception()
        elif (opt in ("--maxCorr")):
            try:
                maxCorr = np.float(value)
                if (maxCorr > 1 or maxCorr < 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : maxCorr must be a float between 0 and 1, not '" + value + "'.\n")
                raise Exception()
        elif (opt in ("--minCorr")):
            try:
                maxCorr = np.float(value)
                if (maxCorr > 1 or maxCorr < 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : minCorr must be a float between 0 and 1, not '" + value + "'.\n")
                raise Exception()
        elif (opt in ("--nogender")):
            nogender = True
        elif (opt in ("--figure")):
            figure = True
        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()
    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        sys.exit("ERROR : You must use --counts.\n" + usage)
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
    logger.info("starting to work")
    startTime = time.time()

    # parse counts from TSV to obtain :
    # - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
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
    # Fragment Per Million normalisation:
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    # this normalisation allows the samples to be compared
    # - countsNorm (np.ndarray[float]): normalised counts of countsArray same dimension
    # for arrays in input/output: NbExons*NbSOIs
    try:
        countsNorm = mageCNV.normalisation.FPMNormalisation(countsArray)
    except Exception:
        logger.error("FPMNormalisation failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done fragments counts normalisation, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Quality control:
    ##################
    # The validity of a sample is evaluated according to its fragment coverage profile.
    # If the profile does not allow to distinguish between poorly covered and covered exons,
    # it's assigned invalid status.
    # - validSampQC (np.array[int]): validity status for each sample passed quality
    #   control (1: valid, 0: invalid), dim = NbSOIs
    # - exonsFewFPM (list[int]): exons indexes with little or no coverage common
    #   to all samples passing quality control
    try:
        if figure:
            outputFile = os.path.join(outFolder, "CoverageProfilChecking_" + str(len(SOIs)) + "samps.pdf")
            (validSampQC, exonsFewFPM) = mageCNV.qualityControl.SampsQC(countsNorm, SOIs, outputFile)
        else:
            (validSampQC, exonsFewFPM) = mageCNV.qualityControl.SampsQC(countsNorm, SOIs)
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
    # filtering the coverage data to recover valid samples and covered exons
    # - validCounts (np.ndarray[float]): normalized fragment counts for exons covered
    # for all samples that passed quality control
    validCounts = np.delete(countsNorm, np.where(validSampQC == 0)[0], axis=1)
    validCounts = np.delete(validCounts, exonsFewFPM, axis=0)

    ###########################
    #### no gender discrimination
    # clustering algorithm direct application
    if nogender:
        logger.info("### Samples clustering:")

        try:
            if figure:
                outputFile = os.path.join(outFolder, "Dendogram_" + str(len(SOIs)) + "Samps_FullChrom.png")
                # applying hierarchical clustering and obtaining 3 outputs:
                # - clusters (np.ndarray[int]): standardized clusterID for each SOIs
                # - ctrls (list[str]): controls clusterID delimited by "," for each SOIs
                # - validSampClust (np.ndarray[boolean]): validity status for each SOIs
                # validated by quality control after clustering (1: valid, 0: invalid)
                (clusters, ctrls, validSampClust) = mageCNV.clustering.clustersBuilds(validCounts, maxCorr, minCorr, minSamps, outputFile)
            else:
                (clusters, ctrls, validSampClust) = mageCNV.clustering.clustersBuilds(validCounts, maxCorr, minCorr, minSamps)

        except Exception:
            logger.error("clusterBuilding failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done clusterisation in %.2f s", thisTime - startTime)
        startTime = thisTime

        #####################################################
        # print results
        mageCNV.clustering.printClustersFile(nogender, SOIs, validSampQC, validSampClust, clusters, ctrls, outFolder)

        thisTime = time.time()
        logger.debug("Done printing results, in %.2f s", thisTime - startTime)
        logger.info("ALL DONE")

    ###########################
    #### gender discrimination
    # avoid grouping Male with Female which leads to dubious CNV calls
    else:

        # - exonsToKeep (list of list[str,int,int,str]): contains information about the covered exons
        exonsToKeep = [val for i, val in enumerate(exons) if i not in exonsFewFPM]

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

        #####################################################
        # Get Autosomes Clusters
        ##################
        logger.info("### Autosomes, sample clustering:")
        try:
            # cutting normalized count data according to autosomal exons
            # - gonoIndexFlat (np.ndarray[int]): flat gonosome exon indexes list
            gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist])
            # - autosomesFPM (np.ndarray[float]): normalized fragment counts for valid samples, exons covered
            # in autosomes
            autosomesFPM = np.delete(validCounts, gonoIndexFlat, axis=0)
            
            if figure:
                outputFile = os.path.join(outFolder, "Dendogram_" + str(autosomesFPM.shape[1]) + "Samps_autosomes.png")
                (clusters, ctrls, validSampClust) = mageCNV.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps, outputFile)
            else:
                (clusters, ctrls, validSampClust) = mageCNV.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps)
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
        logger.info("### Gonosomes, sample clustering:")
        try:
            if figure:
                # applying hierarchical clustering and obtaining 4 outputs:
                # - clusters (np.ndarray[int]): standardized clusterID for each SOIs
                # - ctrls (list[str]): controls clusterID delimited by "," for each SOIs
                # - validSampsClust (np.ndarray[boolean]): validity status for each SOIs
                # validated by quality control after clustering (1: valid, 0: invalid)
                # - genderPred (list[str]): genderID delimited for each SOIs (e.g: "M" or "F")
                (clustersG, ctrlsG, validSampsClustG, genderPred) = mageCNV.clustering.GonosomesClustersBuilds(genderInfo, validCounts, gonoIndex, maxCorr, minCorr, minSamps, figure, outFolder)
            else:
                (clustersG, ctrlsG, validSampsClustG, genderPred) = mageCNV.clustering.GonosomesClustersBuilds(genderInfo, validCounts, gonoIndex, maxCorr, minCorr, minSamps)
        except Exception:
            logger.error("clusterBuilds for gonosomes failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done samples clustering for gonosomes : in %.2f s", thisTime - startTime)
        startTime = thisTime

        #####################################################
        # print results
        ##################
        mageCNV.clustering.printClustersFile(nogender, SOIs, validSampQC, validSampClust, clusters, ctrls, outFolder, validSampsClustG, clustersG, ctrlsG, genderPred)

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
