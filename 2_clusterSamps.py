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
# import sklearn submodule for  Kmeans calculation
import sklearn.cluster

####### MAGE-CNV modules
import mageCNV.countsFile
import mageCNV.normalisation
import mageCNV.clustering
import mageCNV.genderDiscrimination

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
    minDist = 0.05 # 95% pearson correlation
    maxDist = 0.15 # 85% pearson correlation
    # boolean args with False status by default
    nogender = False
    figure = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and 
forms the reference clusters for the call. 
By default, separation of autosomes ("A") and gonosomes ("G") for clustering, to avoid 
bias (chr accepted: X, Y, Z, W).
Results are printed to stdout folder:
- a TSV file format, describe the clustering results, dim = NbSOIs*8 columns:
    1) "sampleID": name of interest samples [str],
    2) "clusterID_A": clusters identifiers [int] obtained through the normalized fragment 
                      counts of exons on autosomes, 
    3) "controlledBy_A": clusters identifiers controlling the sample cluster [str], a 
                         comma-separated string of int values (e.g "1,2"). 
                         If not controlled empty string.
    4) "validitySamps_A": a boolean specifying if a sample is dubious(0) or not(1)[int].
                          This score set to 0 in the case the cluster formed is validated
                          and does not have a sufficient number of individuals.
    5) "genderPreds": a string "M"=Male or "F"=Female deduced by kmeans,
The columns 6, 7 and 8 are the same as 2, 3 and 4 but are specific to gonosomes.
In case the user does not want to discriminate between genders, the output tsv will contain
the format of the first 4 columns for all chromosomes.
- one or more png's illustrating the clustering performed by dendograms. [optionnal]
    Legend : solid line = target clusters , thin line = control clusters
    The clusters appear in decreasing order of distance (1-|r|).

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns 
                   hold the fragment counts.
   --out[str]: acces path to a pre-existing folder to save the output files
   --minSamps [int]: samples minimum number for a cluster creation,
                  default : """+str(minSamps)+""".
   --minDist [float]: minimum distance (1-|pearson correlation|) threshold to start the 
                      cluster formation, default : """+str(minDist)+""".   
   --maxDist [float]: maximum distance (1-|pearson correlation|) threshold to stop the 
                      cluster formation, default : """+str(maxDist)+""".
   --nogender[optionnal]: no autosomes and gonosomes discrimination for clustering. 
   --figure[optionnal]: make dendogram(s) that will be present in the output in png format."""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","out=","minSamps=","minDist=","maxDist=","nogender","figure"])
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : "+e.msg+". Try "+scriptName+" --help\n")
        raise Exception()

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            raise Exception()       
        elif (opt in ("--counts")):
            countsFile = value
            if (not os.path.isfile(countsFile)):
                sys.stderr.write("ERROR : countsFile "+countsFile+" doesn't exist.\n")
                raise Exception()    
        elif (opt in ("--out")):
            outFolder = value
            if (not os.path.isdir(outFolder)):
                sys.stderr.write("ERROR : outFolder "+outFolder+" doesn't exist.\n")
                raise Exception()
        elif (opt in ("--minSamps")):
            try:
                minSamps = np.int(value)
                if (minSamps<0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : minSamps must be a non-negative integer, not '"+value+"'.\n")
                raise Exception()
        elif (opt in ("--minDist")):
            try:
                minDist = np.float(value)
                if (minDist>1 or minDist<0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : minDist must be a float between 0 and 1, not '"+value+"'.\n")
                raise Exception()
        elif (opt in ("--maxDist")):
            try:
                maxDist = np.float(value)
                if (maxDist>1 or maxDist<0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : maxDist must be a float between 0 and 1, not '"+value+"'.\n")
                raise Exception()
        elif (opt in ("--nogender")):
            nogender = True
        elif (opt in ("--figure")):
            figure = True
        else:
            sys.stderr.write("ERROR : unhandled option "+opt+".\n")
            raise Exception()


    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        sys.exit("ERROR : You must use --counts.\n"+usage)

    # AOK, return everything that's needed
    return(countsFile, outFolder, minSamps, minDist, maxDist, nogender, figure)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    scriptName = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want scriptName rather than 'root'
    logger = logging.getLogger(scriptName)

    ################################################
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (countsFile, outFolder, minSamps, minDist, maxDist, nogender, figure) = parseArgs(argv) 

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
    logger.debug("Done parsing countsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Normalisation:
    ##################  
    # allocate a float numpy array countsNorm and populate it with normalised counts of countsArray
    # same dimension for arrays in input/output: NbExons*NBSOIs
    # Fragment Per Million normalisation
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    try :
        countsNorm = mageCNV.normalisation.FPMNormalisation(countsArray)
    except Exception:
        logger.error("FPMNormalisation failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done fragments counts normalisation, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Clustering:
    ####################
    # case where no discrimination between autosomes and gonosomes is made
    # direct application of the clustering algorithm
    if nogender:
        logger.info("### Sample clustering:")
        try: 
            outputFile=os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_FullChrom.png")
            # clusters: an int numpy array containing standardized clusterID for each sample
            # ctrls: a str list containing controls clusterID delimited by "," for each sample 
            # validityStatus: a boolean numpy array containing the validity status for each sample (1: valid, 0: invalid)
            (clusters, ctrls, validityStatus) = mageCNV.clustering.clustersBuilds(countsNorm, minDist, maxDist, minSamps, figure, outputFile)
        except Exception: 
            logger.error("clusterBuilding failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done clusterisation in %.2f s", thisTime-startTime)
        startTime = thisTime

        #####################################################
        # print results
        mageCNV.clustering.printClustersFile(SOIs, clusters, ctrls, validityStatus, outFolder, nogender)

        thisTime = time.time()
        logger.debug("Done printing results, in %.2f s", thisTime-startTime)
        logger.info("ALL DONE")

    ###################  
    # cases where discrimination is made 
    # avoid grouping Male with Female which leads to dubious CNV calls 
    else:
        try: 
            # gonoIndex: is a dictionary where key=GonosomeID(e.g 'chrX')[str], 
            # value=list of gonosome exon index [int].
            # genderInfos: is a str list of lists, contains informations for the gender
            # identification, ie ["gender identifier","particular chromosome"].
            (gonoIndex, genderInfo) = mageCNV.genderDiscrimination.getGenderInfos(exons)
        except Exception: 
            logger.error("getGenderInfos failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime-startTime)
        startTime = thisTime

        # cutting normalized count data according to autosomal or gonosomal exons
        # flat gonosome index list
        gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist]) 
        autosomesFPM = np.delete(countsNorm,gonoIndexFlat,axis=0)
        gonosomesFPM = np.take(countsNorm,gonoIndexFlat,axis=0)

        #####################################################
        # Get Autosomes Clusters
        ##################
        # Application of hierarchical clustering on autosome count data to obtain :
        logger.info("### Autosomes, sample clustering:")
        try :
            outputFile = os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_autosomes.png")
            # resClusteringAutosomes: a 2D numpy array with different columns typing, 
            # extract clustering results based on normalised fragment count from exons overlapping autosomes
            # dim= NbSOIs*4 columns: 
            # 1) SOIs Names [str], 2)clusterID [int], 3)clusterIDToControl [str], 4) Sample validity for calling [int]
            (clusters, ctrls, validityStatus) = mageCNV.clustering.clustersBuilds(autosomesFPM, minDist, maxDist, minSamps, figure, outputFile)
        except Exception:
            logger.error("clusterBuilds for autosomes failed")
            raise Exception()

        thisTime = time.time()
        logger.debug("Done samples clustering for autosomes : in %.2f s", thisTime-startTime)
        startTime = thisTime

        #####################################################
        # Get Gonosomes Clusters
        ##################
        logger.info("### Gonosomes, sample clustering:")

        ##################
        # Performs an empirical method (kmeans) to dissociate male and female. 
        # Kmeans with k=2 (always)
        # kmeans: an int 
        kmeans = sklearn.cluster.KMeans(n_clusters=len(genderInfo), random_state=0).fit(gonosomesFPM.T)

        #####################
        # coverage ratio calcul for the different Kmeans groups and on the different gonosomes 
        # can then associate group Kmeans with a gender
        # gender2Kmeans: a str list of genderID (e.g ["M","F"]), the order correspond to KMeans groupID (gp1=M, gp2=F)
        gender2Kmeans = mageCNV.genderDiscrimination.genderAttribution(kmeans, countsNorm,gonoIndex, genderInfo)

        ####################
        # Independent clustering for the two Kmeans groups
        ### To Fill
        # clustersG: an int 1D numpy array, clusterID associated to SOIsIndex
        clustersG = np.zeros(len(SOIs), dtype=np.int)
        # ctrlsG: a str list containing controls clusterID delimited by "," for each SOIs  
        ctrlsG = [""]*len(SOIs)
        # validityStatusG: a boolean numpy array containing the validity status for SOIs (1: valid, 0: invalid)
        validityStatusG = np.ones(len(SOIs), dtype=np.int)
        # genderPred: a str list containing genderID delimited for each SOIs (e.g: "M" or "F")
        genderPred = [""]*len(SOIs)

        for genderGp in range(len(gender2Kmeans)):
            sampsIndexGp=np.where(kmeans.labels_==genderGp)[0]
            gonosomesFPMGp = gonosomesFPM[:,sampsIndexGp]
            try :
                logger.info("### Clustering samples for gender %s",gender2Kmeans[genderGp])
                outputFile = os.path.join(outFolder,"Dendogram_"+str(len(sampsIndexGp))+"Samps_gonosomes_"+gender2Kmeans[genderGp]+".png")
                (tmpClusters, tmpCtrls, tmpValidityStatus)  = mageCNV.clustering.clustersBuilds(gonosomesFPMGp, minDist, maxDist, minSamps, figure, outputFile)
            except Exception:
                logger.error("clusterBuilds for gonosome failed for gender %s",gender2Kmeans[genderGp])
                raise Exception()
            # populate clusterG, ctrlsG, validityStatusG, genderPred
            for index in range(len(sampsIndexGp)):
                clustersG[sampsIndexGp[index]] = tmpClusters[index]
                ctrlsG[sampsIndexGp[index]] = tmpCtrls[index]
                validityStatusG[sampsIndexGp[index]] = tmpValidityStatus[index]
                genderPred[sampsIndexGp[index]] = gender2Kmeans[genderGp]  

        thisTime = time.time()
        logger.debug("Done samples clustering for gonosomes : in %.2f s", thisTime-startTime)
        startTime = thisTime
        
        #####################################################
        # print results
        ##################
        mageCNV.clustering.printClustersFile(SOIs, clusters, ctrls, validityStatus, outFolder, nogender, clustersG, ctrlsG, validityStatusG, genderPred)
        
        thisTime = time.time()
        logger.debug("Done printing results, in %.2f s", thisTime-startTime)
        logger.info("ALL DONE")

####################################################################################
######################################## Main ######################################
####################################################################################

if __name__ =='__main__':
    try:
        main(sys.argv)
    except Exception:
        # whoever raised the exception should have explained it on stderr, here we just die
        exit(1)