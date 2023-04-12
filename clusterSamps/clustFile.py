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
# Perform multiple sanity checks on clustFile as it's subject to several rules
# The cluster IDs (first column) must be in ascending order from 0 to nbClusters.
# If clustFile contains sex predictions, they must be at the end of the file.
# Samples must be the same as those in the countFile.
# Arg:
# - clustsFile [str]: a clusterFile produced by 2_clusterSamps.py, possibly gzipped
# - samples (list[str]): samples names
# Returns a tupple (sampsInClusts, ctrlsInClusts, validClusts, specClusts), each
# variable is created here:
# - sampsInClusts (list of lists[str]): each index in the list corresponds to a clusterID,
# and each sublist is composed of the "samples" indexes in the cluster
# - ctrlsInClusts (list of lists[str]): each index in the list corresponds to a clusterID,
# and each sublist is composed of the control clusterIDs for the cluster.
# It can be empty if not controlled.
# - validClusts (list[int]): each index in the list corresponds to a cluster associated
# with a validity value (0 invalid, 1 valid)
# - specClusts (list[int]): each index in the list corresponds to a cluster associated
# with an analysis status performed (0 autosomes, 1 gonosomes)
def parseClustsFile(clustsFile, samples):
    try:
        if clustsFile.endswith(".gz"):
            clustsFH = gzip.open(clustsFile, "rt")
        else:
            clustsFH = open(clustsFile, "r")
    except Exception as e:
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception('cannot open clustsFile')

    # skip header
    clustsFH.readline()

    # To Fill and returns
    sampsInClusts, ctrlsInClusts, validClusts, specClusts = [[] for _ in range(4)]

    # To fill not returns
    # boolean array to check that all samples in countsFile are in clustFile.
    # This is only done for autosomes since analysis on gonosomes may be missing
    sampsAutoClusts = np.zeros(len(samples)), np.zeros(len(samples))

    for ind, line in enumerate(clustsFH):
        # finding information from the 5 columns
        clusterID, sampsInCluster, controlledBy, validCluster, specifics = line.rstrip().split("\t", maxsplit=4)

        # sanity check
        # the clusterIDs are integers the first = 0 the following ones are
        # incremented by 1, so they must follow the order of the line indexes,
        # if not the case return an exception
        if ind == np.int(clusterID):
            # populate sampsInClusts with sample indexes
            samps = sampsInCluster.split(",")
            sampsIndexes = [i for i in range(len(samples)) if samples[i] in samps]
            sampsInClusts.append(sampsIndexes)

            # populate ctrlsInClusts
            if controlledBy != "":
                ctrlsInClusts.append([int(x) for x in controlledBy.split(",")])
            else:
                ctrlsInClusts.append([])

            # populate validClusts
            validClusts.append(validCluster)

            # populate specClusts and control boolean np.array
            if specifics == "Autosomes":
                specClusts.append(0)
                sampsAutoClusts[sampsIndexes] = 1
            else:
                specClusts.append(1)

        # a gender prediction may have been made that is not useful for calling
        # normally placed in the last lines of the clustering file in case it's
        # not the case the following loop will returns an exception
        elif line.startswith("M") or line.startswith("F"):
            logger.info()
            continue

        else:
            raise Exception("Cluster IDs in clustFile are not ordered from 0 to nClusters, please correct this")

    # sanity check
    if not sampsAutoClusts.all():
        notSampsClusts = np.where(sampsAutoClusts == 0)[0]
        logger.error("The samples: %s are not contained in the clustFile for autosomal analyses", ",".join([samples[i] for i in notSampsClusts]))
        raise Exception("Some samples are not in clustFile for autosomal analyses")

    clustsFH.close()

    return(sampsInClusts, ctrlsInClusts, validClusts, specClusts)


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
