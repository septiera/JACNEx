###############################################################################################
######################################## MAGE-CNV step 2: Normalisation & clustering ##########
###############################################################################################
# Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and forms the reference 
# clusters for the call. 
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
import mageCNV.clustering


################################################################################################
######################################## Main ##################################################
################################################################################################
def main():
    scriptName = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want scriptName rather than 'root'
    logger = logging.getLogger(scriptName)

    ##########################################
    # parse user-provided arguments
    # mandatory args
    countsFile = ""
    outFolder = ""
    # optionnal arguments
    # default values are fixed
    minSamples = 20
    minDist = 0.05
    maxDist = 0.15
    nogender = False
    figure = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and forms the reference 
clusters for the call. 
By default, separation of autosomes and gonosomes (chr accepted: X, Y, Z, W) for clustering, to avoid bias.
Results are printed to stdout folder:
- a TSV file format, describe the clustering results, dim = NbSOIs*8 columns:
    1) "sampleID": a string for sample of interest full name,
    2) "clusterID_A": an int for the cluster containing the sample for autosomes (A), 
    3) "controlledBy_A": an int list of clusterID controlling the current cluster for autosomes,
    4) "validitySamps_A": a boolean specifying if a sample is dubious(0) or not(1) for the calling step, for autosomes.
                          This score set to 0 in the case the cluster formed is validated and does not have a sufficient 
                          number of individuals.
    5) "genderPreds": a string "M"=Male or "F"=Female deduced by kmeans,
    6) "clusterID_G": an int for the cluster containing the sample for gonosomes (G), 
    7) "controlledBy_G":an int list of clusterID controlling the current cluster ,
    8) "validitySamps_G": a boolean specifying if a sample is dubious(0) or not(1) for the calling step, for gonosomes.
- one or more png's illustrating the clustering performed by dendograms. [optionnal]
    Legend : solid line = target clusters , thin line = control clusters
    The clusters appear in decreasing order of distance (1-|r|).

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns 
                   hold the fragment counts.
   --out[str]: pre-existing folder to save the output files
   --minSamples [int]: an integer indicating the minimum sample number to create a reference cluster for autosomes,
                  default : """+str(minSamples)+""".
   --minDist [float]: is a float variable, sets a minimum distance threshold (1-|r|) for the formation of the first clusters,
                  default : """+str(minDist)+""".   
   --maxDist [float]: is a float variable, it's the maximal distance to concidered,
                  default : """+str(maxDist)+""".
   --nogender[optionnal]: no autosomes and gonosomes discrimination for clustering. 
                  output TSV : dim= NbSOIs*4 columns, ["sampleName", "clusterID", "controlledBy", "validitySamps"]
   --figure[optionnal]: make one or more dendograms that will be present in the output in png format."""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","out=","minSamples=","minDist=","maxDist=","nogender","figure"])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
            if (not os.path.isfile(countsFile)):
                sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif (opt in ("--out")):
            outFolder = value
            if (not os.path.isdir(outFolder)):
                sys.exit("ERROR : outFolder "+outFolder+" doesn't exist. Try "+scriptName+" --help.\n")
        elif (opt in ("--minSamples")):
            try:
                minSamples = np.int(value)
            except Exception as e:
                logger.error("ERROR : minSamples "+str(minSamples)+" conversion to int failed : "+e)
                sys.exit(1)
            if (minSamples<0):
                sys.exit("ERROR : minSamples "+str(minSamples)+" must be a positive int. Try "+scriptName+" --help.\n")
        elif (opt in ("--minDist")):
            try:
                minDist = np.float(value)
            except Exception as e:
                logger.error("ERROR : minDist "+str(minDist)+" conversion to float failed : "+e)
                sys.exit(1)
            if (minDist>1 or minDist<0):
                sys.exit("ERROR : minDist "+str(minDist)+" must be must be between 0 and 1. Try "+scriptName+" --help.\n")
        elif (opt in ("--maxDist")):
            try:
                maxDist = np.float(value)
            except Exception as e:
                logger.error("ERROR : maxDist "+str(maxDist)+" conversion to float failed : "+e)
                sys.exit(1)
            if (maxDist>1 or maxDist<0):
                sys.exit("ERROR : maxDist "+str(maxDist)+" must be must be between 0 and 1. Try "+scriptName+" --help.\n")
        elif (opt in ("--nogender")):
            nogender = True
        elif (opt in ("--figure")):
            figure = True
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

   #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        sys.exit("ERROR : You must use --counts.\n"+usage)

    ######################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    # parse counts from TSV to obtain :
    # - a list of exons same as returned by processBed, ie each
    #    exon is a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
    #    copied from the first 4 columns of countsFile, in the same order
    # - the list of sampleIDs (ie strings) copied from countsFile's header
    # - an int numpy array, dim = len(exons) x len(samples)
    try:
        exons, SOIs, countsArray = mageCNV.countsFile.parseCountsFile(countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed - %s", e)
        sys.exit(1)

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Normalisation:
    ##################  
    # allocate countsNorm and populate it with normalised counts of countsArray
    # same dimension for arrays in input/output: exonIndex*sampleIndex
    # Fragment Per Million normalisation
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    try :
        countsNorm = mageCNV.normalisation.FPMNormalisation(countsArray)
    except Exception as e:
        logger.error("FPMNormalisation failed - %s", e)
        sys.exit(1)
    thisTime = time.time()
    logger.debug("Done FPM normalisation, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Clustering:
    ####################
    # case where no discrimination between autosomes and gonosomes is made
    # direct application of the clustering algorithm
    if nogender:
        try: 
            outputFile=os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_FullChrom.png")
            resClustering = mageCNV.clustering.clustersBuilds(countsNorm, SOIs, minDist, maxDist, minSamples, figure, outputFile)
        except Exception as e: 
            logger.error("clusterBuilding failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.debug("Done clusterisation in %.2f s", thisTime-startTime)
        startTime = thisTime

        #####################################################
        # print results
        mageCNV.clustering.printClustersFile(resClustering,outFolder)
        thisTime = time.time()
        logger.debug("Done printing results, in %.2f s", thisTime-startTime)
        logger.info("ALL DONE")

    ###################  
    # cases where discrimination is made 
    # avoid grouping Male with Female which leads to dubious CNV calls 
    else:
        #identification of gender and keep exons indexes carried by gonosomes
        try: 
            gonoIndexDict, genderInfoList = mageCNV.clustering.getGenderInfos(exons)
        except Exception as e: 
            logger.error("getGenderInfos failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime-startTime)
        startTime = thisTime

        #division of normalized count data according to autosomal or gonosomal exons
        #flat gonosome index list
        gonoIndex = np.unique([item for sublist in list(gonoIndexDict.values()) for item in sublist]) 
        autosomesFPM = np.delete(countsNorm,gonoIndex,axis=0)

        #####################################################
        # Get Autosomes Clusters
        ##################
        # Application of hierarchical clustering on autosome count data to obtain :
        # - a 2D numpy array with different columns typing, extract clustering results,
        #       dim= NbSOIs*4. columns: ,1) SOIs Names [str], 2)clusterID [int], 
        #       3)clusterIDToControl [str], 4) Sample validity for calling [int] 
        logger.info("### Autosomes, samples clustering:")
        try :
            outputFile=os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_autosomes.png")
            resClusteringAutosomes = mageCNV.clustering.clustersBuilds(autosomesFPM, SOIs, minDist, maxDist, minSamples, figure, outputFile)
        except Exception as e:
            logger.error("clusterBuilds for autosomes failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.info("Done samples clustering for autosomes : in %.2f s", thisTime-startTime)
        startTime = thisTime

        #####################################################
        # Get Gonosomes Clusters
        ##################
        # Different treatment
        logger.info("### Gonosomes, samples clustering:")
        try :
            resClusteringGonosomes=mageCNV.clustering.gonosomeProcessing(countsNorm, SOIs, gonoIndexDict, genderInfoList, minSamples, minDist, maxDist, outFolder, figure)
        except Exception as e:
            logger.error("gonosomeProcessing failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.info("Done samples clustering for gonosomes : in %.2f s", thisTime-startTime)
        startTime = thisTime
        
        #####################################################
        # print results
        ##################
        mageCNV.clustering.printClustersFile(resClusteringAutosomes, outFolder, resClusteringGonosomes)
        thisTime = time.time()
        logger.debug("Done printing results, in %.2f s", thisTime-startTime)
        logger.info("ALL DONE")

if __name__ =='__main__':
    main()