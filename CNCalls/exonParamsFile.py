import logging
import gzip
import numpy as np
import numba

# prevent matplotlib and numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
# parseCNcallsFile:
# Arg:
#   - CNcallsFile [str]: produced by s3_CNCalls.py, possibly gzipped
#   - nbState [int]: number of copy number states
#
# Returns a tuple (exons, samples, CNCallsArray), each is created here and populated
# by parsing CNCallsFile:
#   - exons (list of list[str,int,int,str]): exon is a lists of 4 scalars
#     containing CHR,START,END,EXON_ID copied from the first 4 columns of CNcallsFile,
#     in the same order
#   - samples (list[str]): sampleIDs copied from CNcallsFile's header
#   - CNCallsArray (np.ndarray[float]): dim = len(exons) x (len(samples)*4(CN0,CN1,CN2,CN3+))
def parseCNcallsFile(CNcallsFile, nbState):
    (exons, samples, CNCallsList) = parseCNCallsPrivate(CNcallsFile)
    # callsArray[exonIndex,sampleIndex] will store the specified probabilities
    CNCallsArray = allocateParamsArray(len(exons), len(samples), nbState)
    # Fill callsArray from callsList
    for i in range(len(exons)):
        callsVec2array(CNCallsArray, i, CNCallsList[i])

    return(exons, samples, CNCallsArray)


#############################
# printParamsFile:
# Args:
# - outFile is a filename that doesn't exist, it can have a path component (which must exist),
#   output will be gzipped if outFile ends with '.gz'
# - clust2samps (dict): A dictionary mapping cluster IDs to sample names.
# - expectedColNames (list): A list of column names to be used in the output file.
# - exp_loc[float], exp_scale[float]: parameters for the exponential distribution 
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# - CN2ParamsArray (np.ndarray[floats]): parameters of the Gaussian distribution [loc = mean, scale = stdev]
#                                        and exon filtering status [int] for each cluster
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
    toPrint = "" + "\t" + "" + "\t" + "" + "\t" + "exponential parameters"
    expLine = "\t".join(["{:0.2e}\t{:0.2e}\t-1".format(exp_loc, exp_scale)] * len(clust2samps.keys()))
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
# parseCNCallsPrivate
# Arg:
#  - CNcallsFile [str]: produced by s3_CNCalls.py, possibly gzipped
#
# Returns a tuple (exons, samples, CNCallsList), each is created here
#   - exons (list of list[str,int,int,str]): exon is a lists of 4 scalars
#     containing CHR,START,END,EXON_ID copied from the first 4 columns of CNCallsFile,
#     in the same order
#   - samples (list[str]): sampleIDs copied from CNCallsFile's header
#   - CNCallsList (list of list[float]]): dim = len(exons) x (len(samples)*4(CN0,CN1,CN2,CN3+))
#                                         contains log10-likelihood
def parseCNCallsPrivate(CNCallsFile):
    try:
        if CNCallsFile.endswith(".gz"):
            callsFH = gzip.open(CNCallsFile, "rt")
        else:
            callsFH = open(CNCallsFile, "r")
    except Exception as e:
        logger.error("Opening provided CNCallsFile %s: %s", CNCallsFile, e)
        raise Exception('cannot open CNCallsFile')

    # list of exons to be returned
    exons = []
    # list of calls to be returned
    callsList = []

    # grab samples columns from header
    samplesColname = callsFH.readline().rstrip().split("\t")
    # get rid of exon definition headers "CHR", "START", "END", "EXON_ID"
    del samplesColname[0:4]

    # list of unique sample names
    samples = []
    for i in samplesColname:
        samp = i.split("_")[0]
        if samp not in samples:
            samples.append(samp)

    # populate exons and probabilities from data lines
    for line in callsFH:
        # split into 4 exon definition strings + one string containing all the calls
        splitLine = line.rstrip().split("\t", maxsplit=4)
        # convert START-END to ints and save
        exon = [splitLine[0], int(splitLine[1]), int(splitLine[2]), splitLine[3]]
        exons.append(exon)
        # convert calls to 1D np array and save
        calls = np.fromstring(splitLine[4], dtype=np.float32, sep='\t')
        callsList.append(calls)
    callsFH.close()
    return(exons, samples, callsList)


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
