import logging
import gzip
import numpy as np
import matplotlib.pyplot

# different scipy submodules are used for the application of hierachical clustering
import scipy.cluster.hierarchy
import scipy.spatial.distance

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

#############################
# STDZandCheckRes:
# For each cluster ID in clust2Samps, the script creates a sublist with the
# following information:
# The cluster ID
# A string of the samples in that cluster, separated by commas
# (if applicable) a string of the control cluster IDs for that target cluster,
# separated by commas
# A boolean indicating whether or not the cluster has enough samples to be
# considered valid.
# A string indicating the status of the cluster (either "W" for all chromosomes,
# or "A" autosomes or "G" gonosomes for gender-only analysis)
# Also add to the gonosome status whether the cluster contains only "F" females
# or "M" males or both "B".
# The status for these clusters is determined by the predicted genders of the
# samples in the cluster.
# Finally, the script appends a sublist with the samples that have failed quality control
# to clustsResList, and returns the list.
#
# Args:
# - SOIs (list[str]): samples of interest
# - sampsQCfailed (list[int]): samples indexes in SOIs that have failed quality control
# - clust2Samps (dict([int]:list[int])): a dictionary mapping cluster IDs to lists of samples
# indexes in those clusters
# - trgt2Ctrls (dict([int]:list[int])): a dictionary mapping target cluster IDs to lists of
# control cluster IDs.
# - minSamps [int]: the minimum number of samples required for a cluster to be considered valid
# - nogender [boolean]: indicates whether or not gender is being considered in the analysis
# - clust2SampsGono (optional dict([int]: list[int])) : a dictionary mapping cluster IDs to lists
# of samples in those clusters for gender-only analysis
# - trgt2CtrlsGono (optional dict([int]: list[int])) : a dictionary mapping target cluster IDs
# to lists of control cluster IDs for gender-only analysis.
# - kmeans (optional list[int]): for each SOIs indexes attribution of kmeans group (0 or 1),
# indicating which cluster each sample belongs to in a gender-only analysis.
# - sexePred (optional list[int]): predicted genders for each sample.
#
# Returns:
# - clustsResList (list [str]): printable list of cluster information.

def STDZandCheckRes(SOIs, sampsQCfailed, clust2Samps, trgt2Ctrls, minSamps, nogender, clust2SampsGono=None, trgt2CtrlsGono=None, kmeans=None, sexePred=None):
    # extract valid sample name and invalid sample name with list comprehension.
    vSOIs = [SOIs[i] for i in range(len(SOIs)) if i not in sampsQCfailed]
    SOIs_QCFailed = [SOIs[i] for i in range(len(SOIs)) if i in sampsQCfailed]

    clustsResList = []
    # Creating a unified list of cluster IDs, combining both clust2Samps and clust2SampsGono if gender is being considered
    clustIDList = list(clust2Samps.keys()) + (list(clust2SampsGono.keys()) if not nogender else [])

    for i in range(len(clustIDList)):
        # Initialize an empty sublist to store the information for each cluster
        listOflist = [""] * 5
        listOflist[0] = i
        if i < len(clust2Samps.keys()):
            # Get the samples in the current cluster and join them into a string separated by commas
            listOflist[1] = ",".join([vSOIs[i] for i in clust2Samps[clustIDList[i]]])
            if clustIDList[i] in trgt2Ctrls.keys():
                # Get the control clusters for the current target cluster and join their IDs into a string separated by commas
                listOflist[2] = ",".join(str(clustIDList.index(i)) for i in trgt2Ctrls[clustIDList[i]])

            if len(clust2Samps[clustIDList[i]]) < minSamps and listOflist[2] == "":
                # If the cluster does not have enough samples and has no control clusters, mark it as invalid
                listOflist[3] = 0
                logger.warning("cluster N°%s : does not contain enough sample to be valid (%s)", i, len(clust2Samps[clustIDList[i]]))
            else:
                listOflist[3] = 1

            if nogender:
                listOflist[4] = "W"
            else:
                listOflist[4] = "A"
        else:
            # Get the samples in the current cluster and join them into a string separated by commas
            listOflist[1] = ",".join([vSOIs[i] for i in clust2SampsGono[clustIDList[i]]])
            if clustIDList[i] in trgt2CtrlsGono.keys():
                # Get the control clusters for the current target cluster and join their IDs into a string separated by commas
                listOflist[2] = ",".join(str(clustIDList[len(clust2Samps):].index(i) + len(clust2Samps))
                                         for i in trgt2CtrlsGono[clustIDList[i]])
            if len(clust2SampsGono[clustIDList[i]]) < minSamps and listOflist[2] == "":
                # If the cluster does not have enough samples and has no control clusters, mark it as invalid
                listOflist[3] = 0
                logger.warning("cluster N°%s : does not contain enough sample to be valid (%s)", i, len(clust2SampsGono[clustIDList[i]]))
            else:
                listOflist[3] = 1
            # Get the predicted gender of the samples in the current cluster and check if they are all the same
            status = list(set([kmeans[i] for i in clust2SampsGono[clustIDList[i]]]))
            if len(status) == 1:
                # If the samples all have the same predicted gender, add it to the sublist
                listOflist[4] = "G_" + sexePred[status[0]]
            else:
                # If the samples have different predicted genders, mark it as "G_B" (Gonosome, both)
                listOflist[4] = "G_B"
        # Add the sublist to the final list of clusters
        clustsResList.append(listOflist)
    clustsResList.append(["Samps_QCFailed", ",".join(SOIs_QCFailed), "", "", ""])
    return(clustsResList)

