import logging
import gzip
import numpy as np
import numba

import clusterSamps.clustFile

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
# extractCNCallsFromPrev
# Args:
#   - exons (list of list[str,int,int,str]): exon definitions, padded and sorted
#   - samples (list[str]): sample of interest names
#   - clusts2Samps (dict[str, List[int]]): key: clusterID , value: samples index list
#   - prevCNCallsFile [str]: a CNCalls file (possibly gzipped) produced by printCNCallsFile
#     for some samples (hopefully some of the samples), using the same exon definitions
#     as in 'exons', if there is one; or '' otherwise
#   - prevClustsFile [str]: a cluster definition file (possibly gzipped) produced by
#   s2_clusterSamps same timestamp as prevCNCallsFile
#
# Returns a tuple (callsArray, callsFilled), each is created here:
#   - callsArray (np.ndarray[floats]): dim = NbExons x (NbSamples*4), initially -1
#   - callsFilled (np.ndarray[bool]): dim = NbClusters, initially all-False
#
# If prevCNCallsFile=='' return the callsArray all-1 and callsFilled all-False arrays
# Otherwise:
# -> for any sample present in both prevCNCallsFile, prevClustFile and samples, and the
# cluster definition not changes fill the sample's columns in CNcallsArray by
# copying data from prevCNCallsFile, and set callsFilled[sample] to True
def extractCNCallsFromPrev(exons, samples, clusters, prevCNCallsFile, prevClustsFile):
    # numpy arrays to be returned:
    # callsArray[exonIndex,sampleIndex] will store the specified probabilities
    # for each copy number type (CN0,CN1,CN2,CN3+)
    callsArray = allocateCNCallsArray(len(exons), len(samples))
    # callsFilled: same size and order as "clusters", value will be set
    # to True if the cluster remains identical 
    callsFilled = np.zeros(len(clusters), dtype=bool)

    if (prevClustsFile != ''):
        ###################################
        # we have a prevCalls file, parse it
        (prevExons, prevSamples, prevCallsList) = parseCNCallsPrivate(prevCNCallsFile)

        # compare exon definitions
        if (exons != prevExons):
            logger.error("exon definitions disagree between prevCallsFile and countsFile, " +
                         "prevCallsFile cannot be re-used if the BED used for countsFile or padding changed")
            raise Exception('mismatched exons')

        ###################################
        # we have a prevClusts file parse it
        prevClusters = clusterSamps.clustFile.parseClustsFile(prevClustsFile, prevSamples)[0]
        logger.info(prevClusters)

        # fill prev2new to identify SOIs that are in prevClusters:
        # prev2new is a 1D numpy array, size = len(prevSamples), prev2new[prev] is the
        # index in SOIs of sample prevSamples[prev] if it's present, -1 otherwise
        prev2new = np.full(len(prevSamples), -1, dtype=int)
        
        for i, cluster in enumerate(clusters):
            if cluster in prevClusters:
                callsFilled[i] = True
                for j, samp in enumerate(samples):
                    if samp in cluster and samp in prevSamples:
                        prev2new[prevSamples.index(samp)] = j

        # Fill CNcallsArray with prev probabilities data
        for i in range(len(exons)):
            prevCallsVec2CallsArray(callsArray, i, prevCallsList[i], prev2new)

    # return the arrays, whether we had a prevCNCallsFile or not
    return(callsArray, callsFilled)


#############################################################
# parseCNcallsFile:
# Arg:
#   - CNcallsFile [str]: produced by s3_CNCalls.py, possibly gzipped
#
# Returns a tuple (exons, samples, CNCallsArray), each is created here and populated
# by parsing CNCallsFile:
#   - exons (list of list[str,int,int,str]): exon is a lists of 4 scalars
#     containing CHR,START,END,EXON_ID copied from the first 4 columns of CNcallsFile,
#     in the same order
#   - samples (list[str]): sampleIDs copied from CNcallsFile's header
#   - CNCallsArray (np.ndarray[float]): dim = len(exons) x (len(samples)*4(CN0,CN1,CN2,CN3+))
def parseCNcallsFile(CNcallsFile):
    (exons, samples, CNCallsList) = parseCNCallsPrivate(CNcallsFile)
    # callsArray[exonIndex,sampleIndex] will store the specified probabilities
    CNCallsArray = allocateCNCallsArray(len(exons), len(samples))
    # Fill callsArray from callsList
    for i in range(len(exons)):
        callsVec2array(CNCallsArray, i, CNCallsList[i])

    return(exons, samples, CNCallsArray)


#############################
# printCNCallsFile:
# Args:
# - emissionArray (np.ndarray[float]): contain emission probabilities. dim=NbExons* (NbSamples*[CN0,CN1,CN2,CN3+])
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# - samples (list[str]): sample names copied from countsFile's header
# - outFile is a filename that doesn't exist, it can have a path component (which must exist),
#     output will be gzipped if outFile ends with '.gz'
#
# Print this data to outFile as a 'CNCallsFile'
def printCNCallsFile(emissionArray, exons, samples, outFile):
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
    for i in samples:
        for j in range(4):
            toPrint += f"{i}_CN{j}" + "\t"
    toPrint += "\n"
    outFH.write(toPrint)

    #### fill results
    for i in range(len(exons)):
        toPrint = exons[i][0] + "\t" + str(exons[i][1]) + "\t" + str(exons[i][2]) + "\t" + exons[i][3]
        toPrint += calls2str(emissionArray, i)
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
# allocateCNCallsArray:
# Args:
# - numExons, numSamples
# Returns an float array with -1, adapted for storing the probabilities for each
# type of copy number. dim= NbExons x (NbSamples x [CN0, CN1, CN2,CN3+])
def allocateCNCallsArray(numExons, numSamples):
    # order=F should improve performance
    return np.full((numExons, (numSamples * 4)), -1, dtype=np.float32, order='F')


#################################################
# prevCallsVec2CallsArray :
# fill CNcallsArray[exonIndex] with appropriate probabilities from prevCalls, using prev2new to
# know which prev samples (ie columns from prev2new) go where in CNcallsArray[exonIndex].
# This small auxiliary function enables numba optimizations.
# Args:
#   - CNcallsArray (np.ndarray[float]): observed probabilities, dim = NbExons x (NbSamples*4)
#   - exonIndex (int): index of the current exon
#   - prevCalls (np.ndarray[float]): contains the prev probabilities for an exon
#   - prev2new (np.ndarray[int]): size len(prevCalls), prev2new[i] is the column index in callsArray
#   where probabilities for prev sample i (in prevCalls) must be stored, or -1 if sample i must be discarded
@numba.njit
def prevCallsVec2CallsArray(CNcallsArray, exonIndex, prevCalls, prev2new):
    for i in numba.prange(len(prev2new)):
        if prev2new[i] != -1:
            currentColIndex = [x + prev2new[i] * 4 for x in numba.prange(4)]
            prevColIndex = [x + i * 4 for x in numba.prange(4)]
            CNcallsArray[exonIndex, currentColIndex] = prevCalls[prevColIndex]


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
