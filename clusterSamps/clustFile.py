import logging
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
# Returns a tupple (clusts2Samps, clusts2Ctrls, sex2Clust) , each variable is created here:
# - clusts2Samps (dict[str, List[int]]): key: clusterID , value: SOIs list
# - clusts2Ctrls (dict[str, List[str]]): key: clusterID, value: controlsID list
# - sex2Clust (dict[str, list[str]]): key: "A" autosomes or "G" gonosome, value: clusterID list
def parseClustsFile(clustsFile):
    try:
        if clustsFile.endswith(".gz"):
            clustsFH = gzip.open(clustsFile, "rt")
        else:
            clustsFH = open(clustsFile, "r")
    except Exception as e:
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception('cannot open clustsFile')

    # To Fill and returns
    # Initialize the dictionaries to store the clustering information
    clusts2Samps = {}
    clusts2Ctrls = {}
    sex2Clust = {}

    # skip header
    clustsFH.readline()

    for line in clustsFH:
        # last line of the file is associated with the sample that did not pass
        # the quality control only the first two columns are informative
        if line.startswith("Samps_QCFailed"):
            SampsQCFailed = line.rstrip().split("\t", maxsplit=1)[1]
            SampsQCFailed = SampsQCFailed.split(",")
            clusts2Samps["Samps_QCFailed"] = SampsQCFailed
        else:
            # finding information from the 5 columns
            clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus = line.rstrip().split("\t", maxsplit=4)

            """
            #### DEV : For first test step integration of all clusters
            if validCluster == "1":
            """
            # populate clust2Samps
            sampsInCluster = sampsInCluster.split(",")
            clusts2Samps[clusterID] = sampsInCluster

            # populate clusts2Ctrls
            if controlledBy != "":
                clusts2Ctrls[clusterID] = controlledBy.split(",")

            # populate sex2Clust
            if clusterStatus.startswith("A"):
                if "A" in sex2Clust:
                    sex2Clust["A"].append(clusterID)
                else:
                    sex2Clust["A"] = [clusterID]
            elif clusterStatus.startswith("G"):
                if "G" in sex2Clust:
                    sex2Clust["G"].append(clusterID)
                else:
                    sex2Clust["G"] = [clusterID]

    clustsFH.close()
    return(clusts2Samps, clusts2Ctrls, sex2Clust)


#############################
# printClustersFile:
# Args:
# - clustsResList (list of lists[str]): returned by STDZandCheck Function, ie each cluster is a lists
#     of 5 scalars containing [clusterID,Samples,controlledBy,validCluster,status]
# - 'outFile' is a filename that doesn't exist, it can have a path component (which must exist),
#     output will be gzipped if outFile ends with '.gz'
#
# Print this data to outFile as a 'clustsFile' (same format parsed by parseClustsFile).
def printClustersFile(clustsResList, outFile):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open clustersFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open clustersFile')

    toPrint = "clusterID\tSamples\tcontrolledBy\tvalidCluster\tstatus\n"
    outFH.write(toPrint)
    for i in range(len(clustsResList)):
        toPrint = "{}\t{}\t{}\t{}\t{}".format(clustsResList[i][0], clustsResList[i][1],
                                              clustsResList[i][2], clustsResList[i][3],
                                              clustsResList[i][4])
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()
