import logging
import gzip
import numpy as np
import numba

# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
# parseExonParamsFile:
# warning : This function does not explicitly verify that the cluster names
# in exonParamsFile and clustFile are identical because exonParamsFile is
# typically derived from clustFile.
#
# Arg:
#  - exonParamsFile [str]: path to a TSV file containing exon parameters.
#                          produces by s3_exonFilteringAndParams.py.
#  - nbExons [int]: The number of exons.
#  - nbClusters [int]: The number of clusters.
#
# Returns a tupple (exonParamsArray, exp_loc, exp_scale, paramsTitles):
#  - exonParamsArray [list of lists[floats]]: contains the parsed exon parameters for
#                                             each exon and cluster.
#  - exp_loc [float], exp_scale [float: The exponential parameters.
#  - paramsTitles (list[strs]): title of expected columns for each cluster
def parseExonParamsFile(exonParamsFile, nbExons, nbClusters):
    (exonParamsList, exp_loc, exp_scale, paramsTitles) = parseExonParamsPrivate(exonParamsFile)

    nbCol = len(exonParamsList[0]) // nbClusters  # int format necessary for allocateParamsArray
    # Sanity check to ensure the number of columns matches the expected
    if nbCol != len(paramsTitles):
        raise Exception('number of clusters differs between clustFile and exonParamsFile')

    # exonParamsArray[exonIndex, clusterIndex * ["loc", "scale", "filterStatus"]]
    exonParamsArray = allocateParamsArray(nbExons, nbClusters, nbCol)

    # Fill exonParamsArray from exonParamsList
    for i in range(nbExons):
        callsVec2array(exonParamsArray, i, exonParamsList[i])

    return (exonParamsArray, exp_loc, exp_scale, paramsTitles)


#############################
# printParamsFile:
# Args:
# - outFile is a filename that doesn't exist, it can have a path component (which must exist),
#   output will be gzipped if outFile ends with '.gz'
# - clust2samps (dict): A dictionary mapping cluster IDs to sample names.
# - expectedColNames (list): A list of column names to be used in the output file.
# - exp_loc[float], exp_scale[float]: parameters for the exponential distribution.
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# - CN2ParamsArray (np.ndarray[floats]): parameters of the Gaussian distribution [loc = mean, scale = stdev]
#                                        and exon filtering status [int] for each cluster.
#
# Print this data to outFile as a 'ParamsFile'
def printParamsFile(outFile, clust2samps, expectedColNames, exp_loc, exp_scale, exons, CN2ParamsArray):
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
    for i in clust2samps.keys():
        for j in range(len(expectedColNames)):
            toPrint += f"{i}_{expectedColNames[j]}" + "\t"
    toPrint += "\n"
    outFH.write(toPrint)

    #### print first row (exponential parameters)
    toPrint = "" + "\t" + "" + "\t" + "" + "\t" + "exponential parameters"+ "\t"
    expLine = "\t".join(["{:0.4e}\t{:0.4e}\t-1".format(exp_loc, exp_scale)] * len(clust2samps.keys()))
    toPrint += expLine
    toPrint += "\n"
    outFH.write(toPrint)

    #### fill results
    for i in range(len(exons)):
        toPrint = exons[i][0] + "\t" + str(exons[i][1]) + "\t" + str(exons[i][2]) + "\t" + exons[i][3]
        toPrint += calls2str(CN2ParamsArray, i)
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
#   - paramsList (list of list[float]]): dim = nbOfExons * (nbOfClusters * ["loc", "scale", "filterStatus"])
#                                        contains mean, stdev parameters from gaussian distribution and
#                                        exon filter status index from
#                                        ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"].
#  - exp_loc [float], exp_scale [float: The exponential parameters.
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

    # grab header unique title
    header = callsFH.readline().rstrip().split("\t")
    del header[0:4]
    headerTitles = [item.split('_')[-1] for item in header]
    paramsTitles = list(set(headerTitles))

    # grab parameters of the exponential distribution common for all clusters
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

    return (paramsList, exp_loc, exp_scale, paramsTitles)


##############################################################
# allocateParamsArray:
# Args:
# - numExons, numClusters, numCol
#
# Returns an float array with -1, adapted for storing the Gaussian
# parameters [loc = mean, scale = stdev] and exon filtering status [int]
# for each cluster
# dim= NbOfExons * (NbOfClusters * NbOfCol)
def allocateParamsArray(numExons, numClusters, numCol):
    # order=F should improve performance
    return np.full((numExons, (numClusters * numCol)), -1, dtype=np.float64, order='F')


#################################################
# callsVec2array
# fill callsArray[exonIndex] with calls from callsVector (same order)
# Args:
#   - callsArray (np.ndarray[int])
#   - exonIndex (int): is the index of the current exon
#   - callsVector (np.ndarray[float]): contains the probabilities for an exon
@numba.njit
def callsVec2array(callsArray, exonIndex, callsVector):
    for i in numba.prange(len(callsVector)):
        callsArray[exonIndex, i] = callsVector[i]


#############################################################
# calls2str:
# return a string holding the calls from callsArray[exonIndex],
# tab-separated and starting with a tab
def calls2str(callsArray, exonIndex):
    formatted_values = []
    for i in range(callsArray.shape[1]):
        formatted_values.append("{:0.2e}".format(callsArray[exonIndex, i]))
    return "\t" + "\t".join(formatted_values)
