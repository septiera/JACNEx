###############################################################################################
######################################## MAGE-CNV step 2: Sample clustering  ##################
###############################################################################################
# Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and
# forms the reference clusters for the call.
# Prints results in a folder defined by the user.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import numpy as np
import time
import logging
# import sklearn submodule for Kmeans calculation
import sklearn.cluster

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
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and
forms the reference clusters for the call.
By default, separation of autosomes ("A") and gonosomes ("G") for clustering, to avoid
bias (chr accepted: X, Y, Z, W).
XXXXXXXXXXXX
- one or more png's illustrating the clustering performed by dendograms. [optionnal]
    Legend : solid line = control clusters , thin line = target clusters
    The clusters appear in decreasing order of distance (1-|pearson correlation|).

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                   hold the fragment counts. File obtained from 1_countFrags.py
   --out[str]: acces path to a pre-existing folder to save the output files
   --minSamps [int]: samples minimum number for the cluster creation,
                  default : """ + str(minSamps) + """
   --maxCorr [float]: Pearson correlation threshold to start building clusters, default: """ + str(maxCorr) + """
   --minCorr [float]: Pearson correlation threshold to end building clusters, default: """ + str(minCorr) + """
   --nogender[optionnal]: no gender discrimination for clustering
   --figure[optionnal]: make dendogram(s) that will be present in the output in png format\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "out=", "windowSize=", "minSamps=", "maxCorr=", "minCorr=", "nogender", "figure"])
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
        elif (opt in ("--maxCorr")):
            try:
                maxCorr = np.float(value)
                if (maxCorr > 1 or maxCorr < 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : maxCorr must be a float between 0 and 1, not '" + value + "'.\n")
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
    # - exons: a list of exons same as returned by processBed, ie each
    #    exon is a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
    #    copied from the first 4 columns of countsFile, in the same order
    # - SOIs: the list of sampleIDs (ie strings) copied from countsFile's header
    # - countsArray: an int numpy array, dim = NbExons x NbSOIs
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
    # allocate a float numpy array countsNorm and populate it with normalised counts of countsArray
    # same dimension for arrays in input/output: NbExons*NbSOIs
    # Fragment Per Million normalisation:
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
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
    # filtering the coverage data to recover valid samples and covered exons
    validCounts = np.delete(countsNorm, np.where(validSampQC == 0)[0], axis=1)
    validCounts = np.delete(validCounts, exonsFewFPM, axis=0)
    
    ####################
    # case where no discrimination between gender is made
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

"""
    ###################
    # cases where discrimination is made
    # avoid grouping Male with Female which leads to dubious CNV calls
    else:
        try:
            # parse exons to extract information related to the organisms studied and their gender
            # gonoIndex: is a dictionary where key=GonosomeID(e.g 'chrX')[str],
            # value=list of gonosome exon index [int].
            # genderInfos: is a str list of lists, contains informations for the gender
            # identification, ie ["gender identifier","particular chromosome"].
            (gonoIndex, genderInfo) = mageCNV.genderDiscrimination.getGenderInfos(exons)
        except Exception:
            logger.error("getGenderInfos failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime - startTime)
        startTime = thisTime

        # cutting normalized count data according to autosomal or gonosomal exons
        # create flat gonosome index list
        gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist])
        autosomesFPM = np.delete(countsNorm, gonoIndexFlat, axis=0)
        gonosomesFPM = np.take(countsNorm, gonoIndexFlat, axis=0)

        #####################################################
        # Get Autosomes Clusters
        ##################
        logger.info("### Autosomes, sample clustering:")
        try:
            outputFile = os.path.join(outFolder, "Dendogram_" + str(len(SOIs)) + "Samps_autosomes.png")
            # applying hierarchical clustering and obtaining 3 outputs autosomes specific:
            # clusters: an int numpy array containing standardized clusterID for each SOIs
            # ctrls: a str list containing controls clusterID delimited by "," for each SOIs
            # validityStatus: a boolean numpy array containing the validity status for each SOIs (1: valid, 0: invalid)
            (clusters, ctrls, validityStatus) = mageCNV.clustering.clustersBuilds(autosomesFPM, maxCorr, minCorr, minSamps, figure, outputFile)
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

        ##################
        # Performs an empirical method (kmeans) to dissociate male and female.
        # Kmeans with k=2 (always)
        # kmeans: a sklearn.cluster._kmeans.KMeans object where it is possible to extract
        # an int list (kmeans.labels_) indicating the groupID associated to each SOI.
        kmeans = sklearn.cluster.KMeans(n_clusters=len(genderInfo), random_state=0).fit(gonosomesFPM.T)

        #####################
        # calculation of a coverage rate for the different Kmeans groups and on
        # the different gonosomes to associate the Kmeans group with a gender
        # gender2Kmeans: a str list of genderID (e.g ["M","F"]), the order
        # correspond to KMeans groupID (gp1=M, gp2=F)
        gender2Kmeans = mageCNV.genderDiscrimination.genderAttribution(kmeans, countsNorm, gonoIndex, genderInfo)

        ####################
        # Independent clustering for the two Kmeans groups
        ### To Fill
        # clustersG: an int 1D numpy array, clusterID associated for each SOIs
        clustersG = np.zeros(len(SOIs), dtype=np.int)
        # ctrlsG: a str list containing controls clusterID delimited by "," for each SOIs
        ctrlsG = [""] * len(SOIs)
        # validityStatusG: a boolean numpy array containing the validity status for SOIs (1: valid, 0: invalid)
        validityStatusG = np.ones(len(SOIs), dtype=np.int)
        # genderPred: a str list containing genderID delimited for each SOIs (e.g: "M" or "F")
        genderPred = [""] * len(SOIs)

        for genderGp in range(len(gender2Kmeans)):
            sampsIndexGp = np.where(kmeans.labels_ == genderGp)[0]
            gonosomesFPMGp = gonosomesFPM[:, sampsIndexGp]
            try:
                logger.info("### Clustering samples for gender %s", gender2Kmeans[genderGp])
                outputFile = os.path.join(outFolder, "Dendogram_" + str(len(sampsIndexGp)) + "Samps_gonosomes_" + gender2Kmeans[genderGp] + ".png")
                (tmpClusters, tmpCtrls, tmpValidityStatus) = mageCNV.clustering.clustersBuilds(gonosomesFPMGp, maxCorr, minCorr, minSamps, figure, outputFile)
            except Exception:
                logger.error("clusterBuilds for gonosome failed for gender %s", gender2Kmeans[genderGp])
                raise Exception()
            # populate clusterG, ctrlsG, validityStatusG, genderPred
            for index in range(len(sampsIndexGp)):
                clustersG[sampsIndexGp[index]] = tmpClusters[index]
                ctrlsG[sampsIndexGp[index]] = tmpCtrls[index]
                validityStatusG[sampsIndexGp[index]] = tmpValidityStatus[index]
                genderPred[sampsIndexGp[index]] = gender2Kmeans[genderGp]

        thisTime = time.time()
        logger.debug("Done samples clustering for gonosomes : in %.2f s", thisTime - startTime)
        startTime = thisTime

        #####################################################
        # print results
        ##################
        mageCNV.clustering.printClustersFile(SOIs, clusters, ctrls, validityStatus, outFolder, nogender, clustersG, ctrlsG, validityStatusG, genderPred)

        thisTime = time.time()
        logger.debug("Done printing results, in %.2f s", thisTime - startTime)
        logger.info("ALL DONE")
"""

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
