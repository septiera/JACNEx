import gzip
import logging
import numba  # make python faster
import numpy

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
    # dtype=numpy.uint32 should be fast and sufficient
    countsArray = numpy.zeros((len(genomicWindows), len(SOIs)), dtype=numpy.uint32, order='F')
    # countsFilled: same size and order as sampleNames, value will be set
    # to True iff counts were filled from countsFile
    countsFilled = numpy.array([False] * len(SOIs))

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
        prev2new = numpy.full(len(prevSamples), -1, dtype=int)
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
# Parse the counts data in countsFile, normalize (see NOTE) as fragments per
# million (FPM), and return the results separately for exons and intergenic
# pseudo-exons.
# NOTE: for exons on sex chromosomes and intergenic pseudo-exons, FPM normalization
# is performed on all exonic and intergenic counts combined; but for autosome exons
# it is performed taking into account ONLY these autosome exon counts. This
# strategy avoids skewing autosome FPMs in men vs women (due to more reads on chrX
# in women), while preserving ~2x more FPMs in women vs men for chrX exons and
# avoiding huge and meaningless intergenic FPMs (if normalized alone).
#
# Arg:
#   - a countsFile produced by 1_countFrags.py, possibly gzipped
#
# Returns a tuple (samples, autosomeExons, gonosomeExons, intergenics,
#                  autosomeFPMs, gonosomeFPMs, intergenicFPMs),
# each is created here and populated by parsing countsFile:
# -> 'samples' is the list of sampleIDs (strings) copied from countsFile's header
# -> 'autosomeExons', 'gonosomeExons' and 'intergenics' are lists of autosome/gonosome/intergenic
#    (pseudo-)exons as produced by processBed, copied from the first 4 columns of countsFile;
#    EXON_ID is used to decide whether each window is an exon or an intergenic pseudo-exon
# -> 'autosomeFPMs', 'gonosomeFPMs' and 'intergenicFPMs' are numpy 2D-arrays of floats,
#    of sizes [len(autosomeExons) | len(gonosomeExons) | len(intergenics)] x len(samples), holding the
#    FPM-normalized counts for autosome | gonosome exons and intergenic pseudo-exons
def parseAndNormalizeCounts(countsFile):
    (genomicWindows, samples, countsList) = parseCountsFile(countsFile)

    # First pass: identify autosoome/gonosome exons vs intergenic pseudo-exons, populate
    # exons* and intergenics, and calculate sum of autosome/total counts for each sample
    autosomeExons = []
    gonosomeExons = []
    intergenics = []
    # windowType==0 for intergenic pseudo-exons, 1 for gonosome exons, 2 for autosome exons
    windowType = numpy.zeros(len(genomicWindows), dtype=numpy.uint8)
    sumOfCountsAuto = numpy.zeros(len(samples), dtype=numpy.uint32)
    sumOfCountsTotal = numpy.zeros(len(samples), dtype=numpy.uint32)
    sexChroms = sexChromosomes()

    for i in range(len(genomicWindows)):
        if genomicWindows[i][3].startswith("intergenic_"):
            intergenics.append(genomicWindows[i])
            windowType[i] = 0
            sumOfCountsTotal += countsList[i]
        elif genomicWindows[i][0] in sexChroms:
            gonosomeExons.append(genomicWindows[i])
            windowType[i] = 1
            sumOfCountsTotal += countsList[i]
        else:
            autosomeExons.append(genomicWindows[i])
            windowType[i] = 2
            sumOfCountsAuto += countsList[i]

    sumOfCountsTotal += sumOfCountsAuto
    # if any sample has sumOfCounts*==0, replace by 1 to avoid dividing by zero
    sumOfCountsAuto[sumOfCountsAuto == 0] = 1
    sumOfCountsTotal[sumOfCountsTotal == 0] = 1

    # Second pass: populate *FPMs, normalizing the counts on the fly
    autosomeFPMs = numpy.zeros((len(autosomeExons), len(samples)), dtype=numpy.float32)
    gonosomeFPMs = numpy.zeros((len(gonosomeExons), len(samples)), dtype=numpy.float32)
    intergenicFPMs = numpy.zeros((len(intergenics), len(samples)), dtype=numpy.float32)
    # indexes of next auto / gono / intergenic window to populate
    nextAutoIndex = 0
    nextGonoIndex = 0
    nextIntergenicIndex = 0

    for i in range(len(genomicWindows)):
        if windowType[i] == 2:
            autosomeFPMs[nextAutoIndex] = countsList[i] / sumOfCountsAuto
            nextAutoIndex += 1
        elif windowType[i] == 1:
            gonosomeFPMs[nextGonoIndex] = countsList[i] / sumOfCountsTotal
            nextGonoIndex += 1
        else:
            intergenicFPMs[nextIntergenicIndex] = countsList[i] / sumOfCountsTotal
            nextIntergenicIndex += 1

    # scale to FPMs
    autosomeFPMs *= 1e6
    gonosomeFPMs *= 1e6
    intergenicFPMs *= 1e6

    return(samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs, intergenicFPMs)


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
        counts = numpy.fromstring(splitLine[4], dtype=numpy.uint32, sep='\t')
        countsList.append(counts)
    countsFH.close()
    return(genomicWindows, samples, countsList)


#################################################
# prevCountsVec2CountsArray :
# fill countsArray[exonIndex] with appropriate counts from prevCounts, using prev2new to
# know which prev samples (ie columns from prev2new) go where in countsArray[exonIndex].
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


###############################################################################
# sexChromosomes: return a dict whose keys are accepted sex chromosome names,
# and values are 1 for X|Z and 2 for Y|W.
# This covers most species including mammals, birds, fish, reptiles.
# Note that X or Z is present in one or two copies in each individual, and is
# (usually?) the larger of the two sex chromosomes; while Y or W is present
# in 0 or 1 copy and is smaller.
# However interpretation of "having two copies of X|Z" differs: in XY species
# (eg humans) XX is the Female, while in ZW species (eg birds) ZZ is the Male.
def sexChromosomes():
    sexChroms = {"X": 1, "Y": 2, "W": 2, "Z": 1}
    # also accept the same chroms prepended with 'chr'
    for sc in list(sexChroms.keys()):
        sexChroms["chr" + sc] = sexChroms[sc]
    return(sexChroms)
