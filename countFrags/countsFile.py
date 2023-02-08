import numpy as np
import numba  # make python faster
import gzip
import logging


# prevent numba DEBUG messages filling the logs when we are in DEBUG loglevel
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################################################
# extractCountsFromPrev:
# Args:
#   - exons: exon definitions as returned by processBed, padded and sorted
#   - SOIs: a list of samples of interest (ie list of strings)
#   - prevCountsFile: a countsFile (possibly gzipped) produced by printCountsFile
#     for some samples (hopefully some of the SOIs), using the same exon definitions
#     as in 'exons', if there is one; or '' otherwise
#
# Will returns a tuple (countsArray, countsFilled), each is created here:
#   - countsArray is an int numpy array, dim = NbExons x NbSOIs, initially all-zeroes
#   - countsFilled is a 1D boolean numpy array, dim = NbSOIs, initially all-False
#
# If prevCountsFile=='' return the (all-zeroes/all(-False) arrays
# Otherwise:
# -> make sure prevCountsFile was produced with the same BED+padding as exons, else raise exception;
# -> for any sample present in both prevCounts and SOIs, fill the sample's column in
#    countsArray by copying data from prevCounts, and set countsFilled[sample] to True
def extractCountsFromPrev(exons, SOIs, prevCountsFile):
    # numpy arrays to be returned:
    # countsArray[exonIndex,sampleIndex] will store the specified count
    countsArray = allocateCountsArray(len(exons), len(SOIs))
    # countsFilled: same size and order as sampleNames, value will be set
    # to True iff counts were filled from countsFile
    countsFilled = np.array([False] * len(SOIs))

    if (prevCountsFile != ''):
        # we have a prevCounts file, parse it
        (prevExons, prevSamples, prevCountsList) = parseCountsFilePrivate(prevCountsFile)
        # compare exon definitions
        if (exons != prevExons):
            logger.error("exon definitions disagree between prevCountsFile and BED file...\n" +
                         "\tIf the BED file or padding changed, " +
                         "you cannot re-use a previous countsFile: all counts must be recalculated from scratch")
            raise Exception('mismatched exon definitions between prevCountsFile and exons')

        # fill prev2new to identify SOIs that are in prevCountsFile:
        # prev2new is a 1D numpy array, size = len(prevSamples), prev2new[prev] is the
        # index in SOIs of sample prevSamples[prev] if it's present, -1 otherwise
        prev2new = np.full(len(prevSamples), -1, dtype=int)
        # prevIndexes: temp dict, key = sample identifier, value = index in prevSamples
        prevIndexes = {}
        for prevIndex in range(len(prevSamples)):
            prevIndexes[prevSamples[prevIndex]] = prevIndex
        for newIndex in range(len(SOIs)):
            if SOIs[newIndex] in prevIndexes:
                prev2new[prevIndexes[SOIs[newIndex]]] = newIndex
                countsFilled[newIndex] = True

        # Fill countsArray with prev count data
        for i in range(len(exons)):
            prevCountsVec2CountsArray(countsArray, i, prevCountsList[i], prev2new)

    # return the arrays, whether we had a prevCountsFile or not
    return(countsArray, countsFilled)


#############################################################
# parseCountsFile:
# Arg:
#   - a countsFile produced by 1_countFrags.py, possibly gzipped
#
# Returns a tuple (exons, samples, countsArray), each is created here and populated
# by parsing countsFile:
# -> 'exons' is a list of exons same as returned by processBed, ie each
#    exon is a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
#    copied from the first 4 columns of countsFile, in the same order
# -> 'samples' is the list of sampleIDs (ie strings) copied from countsFile's header
# -> 'countsArray' is an int numpy array, dim = len(exons) x len(samples)
def parseCountsFile(countsFile):
    (exons, samples, countsList) = parseCountsFilePrivate(countsFile)
    # countsArray[exonIndex,sampleIndex] will store the specified count
    countsArray = allocateCountsArray(len(exons), len(samples))
    # Fill countsArray from CountsList
    for i in range(len(exons)):
        countsVec2array(countsArray, i, countsList[i])

    return(exons, samples, countsArray)


#############################################################
# printCountsFile:
# Args:
#   - 'exons' is a list of exons same as returned by processBed, ie each exon is a lists
#     of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
#   - 'samples' is a list of sampleIDs
#   - 'countsArray' is an int numpy array, dim = len(exons) x len(samples)
#   - 'outFile' is a filename that doesn't exist, it can have a path component (which must exist),
#      output will be gzipped if outFile ends with '.gz'
#
# Print this data to outFile as a 'countsFile' (same format parsed by extractCountsFromPrev).
def printCountsFile(exons, samples, countsArray, outFile):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open outFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open outFile')

    toPrint = "CHR\tSTART\tEND\tEXON_ID\t" + "\t".join(samples) + "\n"
    outFH.write(toPrint)
    for i in range(len(exons)):
        # exon def + counts
        toPrint = exons[i][0] + "\t" + str(exons[i][1]) + "\t" + str(exons[i][2]) + "\t" + exons[i][3]
        toPrint += counts2str(countsArray, i)
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################################################
# parseCountsFilePrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Arg:
#   - a countsFile produced by 1_countFrags.py, possibly gzipped
#
# Returns a tuple (exons, samples, countsList), each is created here:
# -> 'exons' is a list of exons same as returned by processBed, ie each exon is
#    a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
#    copied from the first 4 columns of countsFile, in the same order
# -> 'samples' is the list of sampleIDs (ie strings) copied from countsFile's header
# -> 'countsList' is a list of len(exons) uint32 numpy arrays of size len(samples), filled
#    with the counts from countsFile
def parseCountsFilePrivate(countsFile):
    try:
        if countsFile.endswith(".gz"):
            countsFH = gzip.open(countsFile, "rt")
        else:
            countsFH = open(countsFile, "r")
    except Exception as e:
        logger.error("Opening provided countsFile %s: %s", countsFile, e)
        raise Exception('cannot open countsFile')

    # list of exons to be returned
    exons = []
    # list of counts to be returned
    countsList = []

    # grab samples from header
    samples = countsFH.readline().rstrip().split("\t")
    # get rid of exon definition headers "CHR", "START", "END", "EXON_ID"
    del samples[0:4]

    # populate exons and counts from data lines
    for line in countsFH:
        # split into 4 exon definition strings + one string containing all the counts
        splitLine = line.rstrip().split("\t", maxsplit=4)
        # convert START-END to ints and save
        exon = [splitLine[0], int(splitLine[1]), int(splitLine[2]), splitLine[3]]
        exons.append(exon)
        # convert counts to 1D np array and save
        counts = np.fromstring(splitLine[4], dtype=np.uint32, sep='\t')
        countsList.append(counts)
    countsFH.close()
    return(exons, samples, countsList)


#############################################################
# allocateCountsArray:
# Args:
#   - numExons, numSamples
# Returns an all-zeroes int numpy array, dim = numExons x numSamples, adapted for
# storing the counts
def allocateCountsArray(numExons, numSamples):
    # order=F should improve performance, since we fill the array one column
    # at a time when parsing BAMs
    # dtype=np.uint32 should be fast and sufficient
    return(np.zeros((numExons, numSamples), dtype=np.uint32, order='F'))


#################################################
# prevCountsVec2CountsArray :
# fill countsArray[exonIndex] with appropriate counts from prevCounts, using prev2new to
# know which prev samples (ie columns from prev2new) go where in countsArray[exonIndex].
# This small auxiliary function enables numba optimizations.
# Args:
#   - countsArray is an int numpy array to populate, dim = NbExons x NbSOIs
#   - exonIndex is the index of the current exon
#   - prevCounts contains the prev counts for exon exonIndex, in a 1D np array
#   - prev2new is a 1D np array of ints of size len(prevCounts), prev2new[i] is
#    the column index in countsArray where counts for prev sample i (in prevCounts) must
#    be stored, or -1 if sample i must be discarded
@numba.njit
def prevCountsVec2CountsArray(countsArray, exonIndex, prevCounts, prev2new):
    for i in numba.prange(len(prev2new)):
        if prev2new[i] != -1:
            countsArray[exonIndex, prev2new[i]] = prevCounts[i]


#################################################
# countsVec2array
# fill countsArray[exonIndex] with counts from countsVector (same order)
# Args:
#   - countsArray is an int numpy array to populate
#   - exonIndex is the index of the current exon
#   - countsVector contains the counts for exon exonIndex, in a 1D np array
@numba.njit
def countsVec2array(countsArray, exonIndex, countsVector):
    for i in numba.prange(len(countsVector)):
        countsArray[exonIndex, i] = countsVector[i]


#################################################
# counts2str:
# return a string holding the counts from countsArray[exonIndex],
# tab-separated and starting with a tab
@numba.njit
def counts2str(countsArray, exonIndex):
    toPrint = ""
    for i in range(countsArray.shape[1]):
        toPrint += "\t" + str(countsArray[exonIndex, i])
    return(toPrint)
