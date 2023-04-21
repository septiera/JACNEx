import numpy as np
import numba  # make python faster
import gzip
import logging


# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################################################
# extractCountsFromPrev:
# Args:
#   - genomicWindows: exon and pseudo-exon definitions as returned by processBed
#   - SOIs: list of samples of interest (ie list of strings)
#   - prevCountsFile: a countsFile (possibly gzipped) produced by printCountsFile for some
#     samples (hopefully some of the SOIs), using the same (pseudo-)exon definitions
#     as in 'genomicWindows', if such a file is available; or '' otherwise
#
# Returns a tuple (countsArray, countsFilled), each is created here:
#   - countsArray is an int numpy array, dim = NbGenomicWindows x NbSOIs, initially all-zeroes
#   - countsFilled is a 1D boolean numpy array, dim = NbSOIs, initially all-False
#
# If prevCountsFile=='' return the (all-zeroes/all-False) arrays
# Otherwise:
# -> make sure prevCountsFile was produced with the same BED+padding as genomicWindows,
#    else raise exception;
# -> for any sample present in both prevCounts and SOIs, fill the sample's column in
#    countsArray by copying data from prevCounts, and set countsFilled[sample] to True
def extractCountsFromPrev(genomicWindows, SOIs, prevCountsFile):
    # numpy arrays to be returned:
    # countsArray[exonIndex,sampleIndex] will store the specified count
    # order=F should improve performance, since we fill the array one column
    # at a time when parsing BAMs
    # dtype=np.uint32 should be fast and sufficient
    countsArray = np.zeros((len(genomicWindows), len(SOIs)), dtype=np.uint32, order='F')
    # countsFilled: same size and order as sampleNames, value will be set
    # to True iff counts were filled from countsFile
    countsFilled = np.array([False] * len(SOIs))

    if (prevCountsFile != ''):
        # we have a prevCounts file, parse it
        (prevGenomicWindows, prevSamples, prevCountsList) = parseCountsFile(prevCountsFile)
        # compare genomicWindows definitions
        if (genomicWindows != prevGenomicWindows):
            logger.error("(pseudo-)exon definitions disagree between prevCountsFile and BED, " +
                         "countsFiles cannot be re-used if the BED file or padding changed")
            raise Exception('mismatched genomicWindows')

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
        for i in range(len(genomicWindows)):
            prevCountsVec2CountsArray(countsArray, i, prevCountsList[i], prev2new)

    # return the arrays, whether we had a prevCountsFile or not
    return(countsArray, countsFilled)


#############################################################
# parseAndNormalizeCounts:
# Parse the counts data in countsFile, normalize it as fragments per million (FPM),
# and return the results separately for exons and intergenic pseudo-exons.
# NOTE: FPM normalization is performed on exon and intergenic counts combined,
# otherwise intergenic FPMs become huge and meaningless.
#
# Arg:
#   - a countsFile produced by 1_countFrags.py, possibly gzipped
#
# Returns a tuple (samples, exons, intergenics, exonCountsFPM, intergenicCountsFPM),
# each is created here and populated by parsing countsFile:
# -> 'samples' is the list of sampleIDs (strings) copied from countsFile's header
# -> 'exons' and 'intergenics' are lists of (pseudo-)exons as produced by processBed,
#    copied from the first 4 columns of countsFile; EXON_ID is used to decide whether
#    each window is an exon or an intergenic pseudo-exon
# -> 'exonCountsFPM' and 'intergenicCountsFPM' are numpy 2D-arrays of floats, of sizes
#    len(exons) x len(samples) and len(intergenics) x len(samples), holding the
#    FPM-normalized counts for exons and intergenic pseudo-exons
def parseAndNormalizeCounts(countsFile):
    (genomicWindows, samples, countsList) = parseCountsFile(countsFile)

    # First pass: identify exons vs intergenic pseudo-exons, populate exons and intergenics,
    # and calculate sum of counts for each sample
    exons = []
    intergenics = []
    isExon = np.zeros(len(genomicWindows), dtype=bool)
    sumOfCounts = np.zeros(len(samples), dtype=np.uint32)

    for i in range(len(genomicWindows)):
        if genomicWindows[i][3].startswith("intergenic_"):
            intergenics.append(genomicWindows[i])
        else:
            exons.append(genomicWindows[i])
            isExon[i] = True
        sumOfCounts += countsList[i]
    # if any sample has sumOfCounts==0, replace by 1 to avoid dividing by zero
    sumOfCounts[sumOfCounts == 0] = 1

    # Second pass: populate exonCountsFPM and intergenicCountsFPM, normalizing
    # the counts on the fly
    exonCountsFPM = np.zeros((len(exons), len(samples)), dtype=np.float32)
    intergenicCountsFPM = np.zeros((len(intergenics), len(samples)), dtype=np.float32)
    # indexes of next exon / intergenic window to populate
    nextExonIndex = 0
    nextIntergenicIndex = 0

    for i in range(len(genomicWindows)):
        if isExon[i]:
            exonCountsFPM[nextExonIndex] = countsList[i] / sumOfCounts
            nextExonIndex += 1
        else:
            intergenicCountsFPM[nextIntergenicIndex] = countsList[i] / sumOfCounts
            nextIntergenicIndex += 1

    # scale to FPMs
    exonCountsFPM *= 1e6
    intergenicCountsFPM *= 1e6

    return(samples, exons, intergenics, exonCountsFPM, intergenicCountsFPM)


#############################################################
# printCountsFile:
# Args:
#   - 'genomicWindows' is a list of exons and pseudo-exons as returned by processBed, ie each
#     (pseudo-)exon is a list of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
#   - 'samples' is a list of sampleIDs
#   - 'countsArray' is an int numpy array, dim = len(genomicWindows) x len(samples)
#   - 'outFile' is a filename that doesn't exist, it can have a path component (which must exist),
#      output will be gzipped if outFile ends with '.gz'
#
# Print this data to outFile as a 'countsFile' (same format parsed by extractCountsFromPrev).
def printCountsFile(genomicWindows, samples, countsArray, outFile):
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
    for i in range(len(genomicWindows)):
        # exon def + counts
        toPrint = genomicWindows[i][0] + "\t" + str(genomicWindows[i][1]) + "\t" + str(genomicWindows[i][2]) + "\t" + genomicWindows[i][3]
        toPrint += counts2str(countsArray, i)
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################################################
# parseCountsFile:
# Arg:
#   - a countsFile produced by 1_countFrags.py, possibly gzipped
#
# Returns a tuple (genomicWindows, samples, countsList), each is created here:
# -> 'genomicWindows' is a list of exons and pseudo-exons as returned by processBed,
#    ie each (pseudo-)exon is a list of 4 scalars (types: str,int,int,str) containing
#    CHR,START,END,EXON_ID copied from the first 4 columns of countsFile, in the same order
# -> 'samples' is the list of sampleIDs (ie strings) copied from countsFile's header
# -> 'countsList' is a list of len(genomicWindows) uint32 numpy arrays of size len(samples), filled
#    with the counts from countsFile
def parseCountsFile(countsFile):
    try:
        if countsFile.endswith(".gz"):
            countsFH = gzip.open(countsFile, "rt")
        else:
            countsFH = open(countsFile, "r")
    except Exception as e:
        logger.error("Opening provided countsFile %s: %s", countsFile, e)
        raise Exception('cannot open countsFile')

    # list of (pseudo-)exons to return
    genomicWindows = []
    # list of counts to return
    countsList = []

    # grab samples from header
    samples = countsFH.readline().rstrip().split("\t")
    # get rid of exon definition headers "CHR", "START", "END", "EXON_ID"
    del samples[0:4]

    # populate genomicWindows and counts from data lines
    for line in countsFH:
        # split into 4 exon definition strings + one string containing all the counts
        splitLine = line.rstrip().split("\t", maxsplit=4)
        # convert START-END to ints and save
        exon = [splitLine[0], int(splitLine[1]), int(splitLine[2]), splitLine[3]]
        genomicWindows.append(exon)
        # convert counts to 1D np array and save
        counts = np.fromstring(splitLine[4], dtype=np.uint32, sep='\t')
        countsList.append(counts)
    countsFH.close()
    return(genomicWindows, samples, countsList)


#################################################
# prevCountsVec2CountsArray :
# fill countsArray[exonIndex] with appropriate counts from prevCounts, using prev2new to
# know which prev samples (ie columns from prev2new) go where in countsArray[exonIndex].
# This small auxiliary function enables numba optimizations.
# Args:
#   - countsArray is an int numpy array to populate, dim = NbGenomicWindows x NbSOIs
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
