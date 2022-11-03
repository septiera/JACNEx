import sys
import numpy as np # numpy arrays
import numba # make python faster
import logging

# set up logger, using inherited config
logger = logging.getLogger(__name__)

#############################################################
################ Function
#############################################################
# parseCountsFile:
# Input:
#   - countsFH is a filehandle open for reading, content is a countsFile 
#     as produced by this program
#   - exons holds the exon definitions, padded and sorted, as returned
#     by processBed but with START and END converted to strings
#   - SOIs is a list of strings: the sample names of interest
#   - countsArray is an all-zeroes int numpy array (dim=NbExons*NbSOIs)
#   - countsFilled is an all-False boolean numpy array, same size as SOIs
#
# -> make sure countsFile was produced with the same BED as exons, else die;
# -> for any sample present in both countsFH and SOIs, fill the sample's
#    column in countsArray by copying data from countsFH, and set
#    countsFilled[sample] to True
def parseCountsFile(countsFH,exons,SOIs,countsArray,countsFilled):
    ######################
    # parse header from (old) countsFile
    oldHeader = countsFH.readline().rstrip().split("\t")
    # ignore exon definition headers "CHR", "START", "END", "EXON_ID"
    del oldHeader[0:4]
    # fill old2new to identify samples of interest that are already in countsFile:
    # old2new is a vector, size is the number of samples in countsFH, value old2new[old] is:
    # the index in SOIs (if present) of the old sample (at index "old" in countsFH 
    #   after discarding exon definitions);
    # -1 otherwise, ie if the old sample at index "old" is absent from SOIs
    old2new = np.full(len(oldHeader), -1, dtype=int)
    # oldIndexes: temp dict, key = sample identifier, value = index in oldHeader
    oldIndexes = {}
    for oldIndex in range(len(oldHeader)):
        oldIndexes[oldHeader[oldIndex]] = oldIndex
    for newIndex in range(len(SOIs)):
        if SOIs[newIndex] in oldIndexes:
            old2new[oldIndexes[SOIs[newIndex]]] = newIndex
            countsFilled[newIndex] = True
    ######################
    # parse data lines from countsFile
    exonIndex = 0
    for line in countsFH:
        # split into 4 exon definition strings + one string containing all the counts
        splitLine=line.rstrip().split("\t",maxsplit=4)
        ####### Compare exon definitions
        if ((splitLine[0] != exons[exonIndex][0]) or
            (splitLine[1] != exons[exonIndex][1]) or
            (splitLine[2] != exons[exonIndex][2]) or
            (splitLine[3] != exons[exonIndex][3])) :
            logger.error("exon definitions disagree between previous countsFile and the provided BED file "+
                         "(after padding and sorting) for exon index %i. If the BED file changed, a previous "+
                         "countsFile cannot be re-used: all counts must be recalculated from scratch", exonIndex)
            sys.exit(1)
        ###### Fill countsArray with old count data
        # convert counts from strings to ints, populating a numpy array (needed for numba)
        oldCounts = np.fromstring(splitLine[4], dtype=np.uint32, sep='\t') 
        oldCount2CountArray(countsArray,exonIndex,oldCounts,old2new)
        ###### next line
        exonIndex += 1

#################################################
# oldCount2CountArray :
# fill countsArray[exonIndex] with appropriate counts from oldCounts, using old2new to
# know which old samples (ie columns from old2new) go where in countsArray[exonIndex].
# This small auxiliary function enables numba optimizations.
# Args:
#   -countsArray is an int numpy array, row exonIndex will be filled  (dim=NbExons*NbSOIs)
#   -exonIndex is the index of the current exon (ie row in countsArray)
#   -oldCounts contains the old counts for exon exonIndex
#   -old2new is a vector of size oldCounts, old2new[i] is the column index in countsArray where
#    counts for old sample i (in oldCounts) must be stored, or -1 if sample i must be discarded
@numba.njit
def oldCount2CountArray(countsArray,exonIndex,oldCounts,old2new):
    for oldIndex in numba.prange(len(old2new)):
            if old2new[oldIndex]!=-1:
                countsArray[exonIndex][old2new[oldIndex]] = oldCounts[oldIndex]
