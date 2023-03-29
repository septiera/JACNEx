import logging
import numpy as np
import gzip

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#####################################
# parseClustsFile
# Arg:
# - clustsFile (str): a clusterFile produced by 2_clusterSamps.py, possibly gzipped
#
# Returns a tupple (clusts2Samps, clusts2Ctrls), each variable is created here:
# - clusts2Samps (dict[str, List[int]]): key: clusterID , value: SOIs list
# - clusts2Ctrls (dict[str, List[str]]): key: clusterID, value: controlsID list
def parseClustsFile(clustsFile, samples):
    try:
        if clustsFile.endswith(".gz"):
            clustsFH = gzip.open(clustsFile, "rt")
        else:
            clustsFH = open(clustsFile, "r")
    except Exception as e:
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception('cannot open clustsFile')

    # To Fill and returns
    samps2Clusters = {}
    clusts2Ctrls = {}

    # skip header
    clustsFH.readline()

    # path of each clusterID
    for line in clustsFH:
        if line.startswith("M") or line.startswith("F"):
            continue
        else:
            # finding information from the 5 columns
            clusterID, sampsInCluster, controlledBy, validCluster, specifics = line.rstrip().split("\t", maxsplit=4)

            # For DEV evaluate small clusters
            # if validCluster == 0:
            #     continue

            sampsInCluster = sampsInCluster.split(",")
            sampsIndInCluster = [i for i in range(len(samples)) if samples[i] in sampsInCluster]

            # populate samps2Clusters
            samps2Clusters[clusterID] = sampsIndInCluster
            if controlledBy != "":
                # populate clusts2Ctrls
                clusts2Ctrls[clusterID] = controlledBy.split(",")

    clustsFH.close()

    return(samps2Clusters, clusts2Ctrls)


#############################
# printClustsFile:
# convert sample indexes to samples names before printing
# clustersID from gonosomes are renamed (+ nbClusterAutosomes)
# Args:
# - autosClusters (list of lists[int]): [clusterID,[Samples],[controlledBy]]
# - gonosClusters (list of lists[int]): can be empty
# - samples (list[str]): samples names
# - sexAssign (list of lists[str]): for each sexe is assigned samples list
# - 'outFile' is a filename that doesn't exist, it can have a path component (which must exist),
#     output will be gzipped if outFile ends with '.gz'
#
# Print this data to outFile as a 'clustsFile' (same format parsed by parseClustsFile).
def printClustsFile(autosClusters, gonosClusters, samples, sexAssign, outFile):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open clustersFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open clustersFile')

    toPrint = "CLUSTER_ID\tSAMPLES\tCONTROLLED_BY\tVALIDITY\tSPECIFICS\n"
    outFH.write(toPrint)

    # browsing the two clustering result tables
    for index in range(2):
        # to give a different name to the gonosomal clusters
        incremName = 0
        if index == 0:
            clustList = autosClusters
            specifics = "Autosomes"
        else:
            incremName = len(autosClusters)
            clustList = gonosClusters
            specifics = "Gonosomes"

        # no print if clustering could not be performed
        if len(clustList) != 0:
            for cluster in range(len(clustList)):
                samplesNames = [samples[j] for j in clustList[cluster][1]]
                toPrint = "{}\t{}\t{}\t{}\t{}".format(str(clustList[cluster][0] + incremName),
                                                      ",".join(samplesNames),
                                                      ",".join([str(j + incremName) for j in clustList[cluster][2]]),
                                                      str(clustList[cluster][3]),
                                                      specifics)
                toPrint += "\n"
                outFH.write(toPrint)

    # sex prediction has been made
    # addition of two lines Male , Female with the list of corresponding samples
    if len(sexAssign) != 0:
        for sexInd in range(len(sexAssign)):
            toPrint = "{}\t{}\t{}\t{}\t{}".format(sexAssign[sexInd][0],
                                                  ",".join(sexAssign[sexInd][1]),
                                                  "",
                                                  "",
                                                  "")
            toPrint += "\n"
            outFH.write(toPrint)

    outFH.close()
