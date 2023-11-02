import logging
import gzip
import numpy as np
import os

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
# parseExonParamsFile:
#
# Arg:
#  - exonParamsFile [str]: path to a TSV file containing exon parameters.
#                          input name of s3_exonCalls.py.
#
# Returns a tupple (exonParamsArray, exp_loc, exp_scale, paramsTitles):
#  - exonMetrics_A [dict]: keys == clusterID and values == np.ndarray[floats]
#                          dim  = NbOfExons * NbOfMetrics, contains the fitting
#                          results of the Gaussian distribution and filters for
#                          exons in autosomes.
#  - exonMetrics_G [dict]: same as exonMetrics_A for gonosomes.
#  - exp_loc [float], exp_scale [float: The half normal parameters.
#  - metricsNames (list[strs]): ["loc","scale","filterStates"]
def parseExonParamsFile(exonParamsFile):
    exonMetrics_A = {}
    exonMetrics_G = {}
    # presence of two results files (gonsomes and autosomes)
    paramsRootFile = os.path.splitext(exonParamsFile)[0]
    paramsRootFile_A = paramsRootFile + "_A.gz"
    paramsRootFile_G = paramsRootFile + "_G.gz"

    (clusterIDs_A, metricsNames_A, paramsList_A, exp_loc_A, exp_scale_A) = parseExonParamsPrivate(paramsRootFile_A)
    (clusterIDs_G, metricsNames_G, paramsList_G, exp_loc_G, exp_scale_G) = parseExonParamsPrivate(paramsRootFile_G)

    exonMetrics_A = fillParamsDict(clusterIDs_A, paramsList_A, metricsNames_A)
    exonMetrics_G = fillParamsDict(clusterIDs_G, paramsList_G, metricsNames_G)

    return (exonMetrics_A, exonMetrics_G, exp_loc_A, exp_scale_A, metricsNames_A)


#############################
# printParamsFile:
# Args:
# - outFile is a filename that doesn't exist, it can have a path component (which must exist),
#   output will be gzipped if outFile ends with '.gz'
# - exonMetrics (dict): A dictionary mapping cluster IDs to np.ndarray of exon metrics results.
# - metricsNames (list[str])
# - exp_loc[float], exp_scale[float]: parameters for the half normal distribution.
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
#
# Print this data to outFile as a 'ParamsFile'
def printParamsFile(outFile, exonMetrics, metricsNames, exp_loc, exp_scale, exons):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open CNCallsFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open CNCallsFile')

    ### header definitions
    toPrint = "CHR\tSTART\tEND\tEXON_ID\t"

    listClustID = sorted(list(exonMetrics.keys()))
    for i in listClustID:
        for j in range(len(metricsNames)):
            toPrint += f"{i}_{metricsNames[j]}" + "\t"
    toPrint += "\n"
    outFH.write(toPrint)

    #### print first row (half normal parameters)
    toPrint = "" + "\t" + "" + "\t" + "" + "\t" + "half normal parameters" + "\t"
    expLine = "\t".join(["{:0.4e}\t{:0.4e}\t-1".format(exp_loc, exp_scale)] * len(exonMetrics.keys()))
    toPrint += expLine
    toPrint += "\n"
    outFH.write(toPrint)

    #### fill results
    for i in range(len(exons)):
        toPrint = exons[i][0] + "\t" + str(exons[i][1]) + "\t" + str(exons[i][2]) + "\t" + exons[i][3]
        for clust in listClustID:
            toPrint += calls2str(exonMetrics[clust], i)
        toPrint += "\n"
        outFH.write(toPrint)

    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################################################
# parseExonParamsPrivate
# Arg:
#  - CNcallsFile [str]: path to a TSV file containing exon parameters.
#                       produces by s3_exonFilteringAndParams.py.
#
# Returns a tuple (paramsList, exp_loc, exp_scale, paramsTitles), each is created here
#   - paramsList (list of list[floats]]): dim = nbOfExons * (nbOfClusters * ["loc", "scale", "filterStatus"])
#                                        contains mean, stdev parameters from gaussian distribution and
#                                        exon filter status index from
#                                        ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"].
#  - exp_loc [float], exp_scale [float: The half normal parameters.
#  - paramsTitles (list[strs]): title of expected columns for each cluster
def parseExonParamsPrivate(exonParamsFile):
    try:
        if exonParamsFile.endswith(".gz"):
            callsFH = gzip.open(exonParamsFile, "rt")
        else:
            callsFH = open(exonParamsFile, "r")
    except Exception as e:
        logger.error("Opening provided CNCallsFile %s: %s", exonParamsFile, e)
        raise Exception('cannot open CNCallsFile')

    # Read the first line of the file and split it into header titles
    header = callsFH.readline().rstrip().split("\t")
    del header[0:4]  # Remove the first four columns

    # Initialize lists to store unique parts
    clusterIDs = []
    paramsTitles = []
    paramsList = []
    exp_loc = 0
    exp_scale = 0

    # Extract unique parts after the second "_" and store them in sets
    if len(header) != 0:
        # Extract unique parts after the second "_" and store them in sets
        for item in header:
            parts = item.split('_', 2)  # Split at most 2 times
            clust = parts[0] + "_" + parts[1]
            metric = parts[2]
            if clust not in clusterIDs:
                clusterIDs.append(clust)
            if metric not in paramsTitles:
                paramsTitles.append(metric)

        # Convert sets to lists
        clusterIDs = list(clusterIDs)
        paramsTitles = list(paramsTitles)

        # grab parameters of the half normal distribution common for all clusters
        expLine = callsFH.readline().rstrip().split("\t")
        exp_loc = np.float64(expLine[4])
        exp_scale = np.float64(expLine[5])

        paramsList = []
        # populate paramsList from data lines
        for line in callsFH:
            # split into 4 exon definition strings + one string containing all the calls
            splitLine = line.rstrip().split("\t", maxsplit=4)
            # not retrieve exon information as it has already been extracted using
            # the countsFile.
            # convert params to 1D np array and save
            params = np.fromstring(splitLine[4], dtype=np.float64, sep='\t')
            paramsList.append(params)
        callsFH.close()      

    return (clusterIDs, paramsTitles, paramsList, exp_loc, exp_scale)


#############################################################
# calls2str:
# return a string holding the calls from callsArray[exonIndex],
# tab-separated and starting with a tab
def calls2str(callsArray, exonIndex):
    formatted_values = []
    for i in range(callsArray.shape[1]):
        formatted_values.append("{:0.2e}".format(callsArray[exonIndex, i]))
    return "\t" + "\t".join(formatted_values)


############################
# fillParamsDict
# Fill a dictionary with exon metrics for each cluster.
#
# Args:
# - clusterIDs (list[strs]): List of cluster IDs.
# - paramsList (list of lists[floats]): List of exon parameters where each sublist represents
#                                       a set of parameters for each exon.
# - metricsNames (list[strs]): List of metric names.
#
# Returns:
# - exonMetrics (dict): keys == cluster IDs; values == 2D NumPy arrays representing exon metrics.
def fillParamsDict(clusterIDs, paramsList, metricsNames):
    numMetrics = len(metricsNames)
    exonMetrics = {}
    if len(paramsList) != 0:
        for clust in clusterIDs:
            # Initialize the dictionary value with a 2D array of -1's using np.full().
            exonMetrics[clust] = np.full((len(paramsList), len(metricsNames)), -1, dtype=np.float64, order='C')

        # Fill the exonMetrics dictionary by iterating over exons and clusters.
        for ei, exon_params in enumerate(paramsList):
            for ci, cluster_id in enumerate(clusterIDs):
                exonMetrics[cluster_id][ei, :] = exon_params[ci * numMetrics: ci * numMetrics + numMetrics]

    return(exonMetrics)
