import sys
import os
import logging


# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#####################################
# parseClustsFile ### !!! TO copy in clustering.py
# extraction of clustering informations
#
# Args:
# - clustsFile (str): a clusterFile produced by 2_clusterSamps.py
# - SOIs (List[str]): samples of interest names list
# Returns a tupple (clusts2Samps,clusts2Ctrls) , each variable is created here:
# - clusts2Samps (Dict[str, List[int]]): key: clusterID , value: samples index list
# - clusts2Ctrls (Dict[str, List[str]]): key: clusterID, value: controlsID list
# - nogender (boolean): identify if a discrimination of gender is made
def parseClustsFile(clustsFile, SOIs):
    try:
        clustsFH = open(clustsFile, "r")
        # Read the first line of the file, which contains the headers, and split it by tab character
        headers = clustsFH.readline().rstrip().split("\t")
    except Exception as e:
        # If there is an error opening the file, log the error and raise an exception
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception("cannot open clustsFile")

    # To Fill and returns
    # Initialize the dictionaries to store the clustering information
    clusts2Samps = {}
    clusts2Ctrls = {}

    # Initialize a flag to indicate whether gender is included in the clustering
    nogender = False

    # clusterFile shows that no gender discrimination is made
    if headers == ["samplesID", "clusterID", "controlledBy", "validitySamps"]:
        for line in clustsFH:
            # Split the line by tab character and assign the resulting
            # values to the following variables:
            # sampleID: the ID of the sample
            # validQC: sample valid status (0: invalid, 1: valid)
            # validClust: cluster valid status (0: invalid, 1: valid)
            # clusterID: clusterID ("" if not validated)
            # controlledBy: a list of cluster IDs that the current cluster is controlled by ("" if not validated)
            sampleID, validQC, validClust, clusterID, controlledBy = line.rstrip().split("\t", maxsplit=4)

            if validClust == "1":
                clusts2Samps.setdefault(clusterID, []).append(SOIs.index(sampleID))
                if controlledBy:
                    clusts2Ctrls[clusterID] = controlledBy.split(",")
            else:
                # If the sample is not valid, log a warning message
                logger.warning(f"{sampleID} dubious sample.")

    # clusterFile shows that gender discrimination is made
    elif headers == ["samplesID", "validQC", "validCluster_A", "clusterID_A",
                     "controlledBy_A", "genderPreds", "validCluster_G",
                     "clusterID_G", "controlledBy_G"]:
        nogender = True
        for line in clustsFH:
            # Split the line by tab character and assign the resulting
            # values to the following variables:
            # duplicate validClust, clusterID, controlledBy columns for autosomes (A) and gonosomes (G)
            # genderPreds: gender "M" or "F" ("" if not validated)
            split_line = line.rstrip("\n").split("\t", maxsplit=8)
            (sampleID, validQC, validClust_A, clusterID_A, controlledBy_A,
             genderPreds, validClust_G, clusterID_G, controlledBy_G) = split_line

            # populate clust2Samps
            # extract only valid samples for autosomes because it's
            # tricky to eliminate samples on their validity associated with gonosomes,
            # especially if one gender has few samples.
            if validClust_A == "1":
                # fill clusterID with fine description (A for autosomes,M for Male, F for Female)
                #################
                # Autosomes
                clusterID_A = f"{clusterID_A}A"
                clusts2Samps.setdefault(clusterID_A, []).append(SOIs.index(sampleID))
                if controlledBy_A:
                    clusts2Ctrls[clusterID_A] = [f"{ctrl}A" for ctrl in controlledBy_A.split(",")]

                #################
                # Gonosomes
                clusterID_G = f"{clusterID_G}{genderPreds}"
                clusts2Samps.setdefault(clusterID_G, []).append(SOIs.index(sampleID))
                if controlledBy_G:
                    clusts2Ctrls[clusterID_G] = [f"{ctrl}{genderPreds}" for ctrl in controlledBy_G.split(",")]
            else:
                logger.warning(f"{sampleID} dubious sample.")
    else:
        logger.error(f"Opening provided clustsFile {clustsFile}: {headers}")
        raise Exception("cannot open clustsFile")

    return(clusts2Samps, clusts2Ctrls, nogender)

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################