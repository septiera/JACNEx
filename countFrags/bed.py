import gzip
import logging
import numpy as np

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################################################
# Parse and process BED file holding exon definitions.
# Args : bedFile == a bed file (with path), possibly gzipped, containing exon definitions
#            formatted as CHR START END EXON_ID
#        padding (int)
#
# Returns a list of [numberOfExons] lists of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID, and where:
# - exons are padded, ie -padding for START (never negative) and +padding for END
# - intergenic regions are introduced
# - exons and intergenic regions are sorted by CHR, then START, then END, then EXON_ID

def processBed(bedFile, padding):
    # list of exons to be returned
    exons = []
    # dictionary for checking that EXON_IDs are unique (key='EXON_ID', value=1)
    exonIDs = {}

    try:
        if bedFile.endswith(".gz"):
            bedFH = gzip.open(bedFile, "rt")
        else:
            bedFH = open(bedFile, "r")
    except Exception as e:
        logger.error("Cannot open BED file - %s", e)
        raise Exception("processBed failed, check log")

    for line in bedFH:
        fields = line.rstrip().split("\t")
        # sanity checks + preprocess START+END
        # need exactly 4 fields
        if len(fields) != 4:
            logger.error("In BED file %s, line doesn't have 4 fields:\n%s",
                         bedFile, line)
            raise Exception("processBed failed, check log")
        # START and END must be ints
        if fields[1].isdigit() and fields[2].isdigit():
            # OK, convert to actual ints and pad (but don't pad to negatives)
            fields[1] = max(int(fields[1]) - padding, 0)
            fields[2] = int(fields[2]) + padding
        else:
            logger.error("In BED file %s, columns 2-3 START-END are not ints in line:\n%s",
                         bedFile, line)
            raise Exception("processBed failed, check log")
        # EXON_ID must be unique
        if fields[3] in exonIDs:
            logger.error("In BED file %s, EXON_ID (4th column) %s is not unique",
                         bedFile, fields[3])
            raise Exception("processBed failed, check log")
        else:
            exonIDs[fields[3]] = 1
        # OK, save exon definition
        exons.append(fields)
    bedFH.close()

    # Done parsing bedFile, sort exons and return
    sortExonsOrBPs(exons)
    # add intergenic windows
    exonsAndInterRegions = addIntergenicWindows(exons)

    return(exonsAndInterRegions)


####################################################
# Sort list of exons (or breakpoints, a very similar data structure):
# an "exon" is a 4-element list [CHR,START,END,EXON_ID],
# a "breakpoint" is a 5-element list [CHR, START, END, CNVTYPE, QNAME].
# In both cases we want to sort by chrom then START then END then <remaining columns>,
# but we want chroms sorted correctly (eg chr2 < chr11, contrary to string sorting).
#
# Returns nothing, sort occurs in-place.
def sortExonsOrBPs(data):
    # key == CHR, value == numeric version of CHR
    chr2num = {}
    # dict to store all non-numeric chromosomes, key is the CHR stripped of 'chr' if
    # present and value is the CHR (e.g 'Y'->'chrY')
    nonNumChrs = {}
    # maxCHR: max int value in CHR column (after stripping 'chr' if present)
    maxCHR = 0

    for thisData in data:
        thisChr = thisData[0]
        if thisChr not in chr2num:
            # strip chr prefix if present
            if thisChr.startswith('chr'):
                chrStripped = thisChr[3:]
            else:
                chrStripped = thisChr

            if chrStripped.isdigit():
                # numeric chrom: populate chr2num
                chrNum = int(chrStripped)
                chr2num[thisChr] = chrNum
                if maxCHR < chrNum:
                    maxCHR = chrNum
            else:
                # non-numeric chromosome: save in nonNumChrs for later
                nonNumChrs[chrStripped] = thisChr
                # also populate chr2num with dummy value, to skip this chr next time we see it
                chr2num[thisChr] = -1

    # map non-numerical chromosomes to maxCHR+1, maxCHR+2 etc
    # first deal with X, Y, M/MT in that order
    for chrom in ["X", "Y", "M", "MT"]:
        if chrom in nonNumChrs:
            maxCHR += 1
            chr2num[nonNumChrs[chrom]] = maxCHR
            del nonNumChrs[chrom]
    # now deal with any other non-numeric CHRs, in alphabetical order
    for chrom in sorted(nonNumChrs):
        maxCHR += 1
        chr2num[nonNumChrs[chrom]] = maxCHR

    # append temp CHR_NUM column to breakpoints
    for thisData in data:
        thisData.append(chr2num[thisData[0]])
    # sort by CHR_NUM then START etc..., last sort on row[4] needed for
    # breakpoints and doesn't hurt for exons
    data.sort(key=lambda row: (row[-1], row[1], row[2], row[3], row[4]))
    # delete the tmp column
    for thisData in data:
        thisData.pop()
    return()


####################################################
# addition of intergenic regions to canonical exons to generate a reliable
# non-coverage profile.
# The aim is to create intergenic windows that do not cover any canonical exon.
# These intergenic windows must be optimally spaced.
# This is achieved by maintaining a natural pattern by calculating the median
# exon length and a distance between exons derived from the largest majority of exons
# (or overlap and or chromosomal difference).
# Arg:
# - exons (list of list[str, int, int, str]): exons definition [CHR, START, END, EXON_ID]
# Returns the same format as the input but with the addition of intergenic regions
# the output remains sorted
def addIntergenicWindows(exons):
    # To Fill and returns
    exonsAndInterRegion = []

    # To Fill not returns
    # list of int containing the lengths of each exon
    lenEx = []

    # numpy array of exon lengths set to zero,
    # if exons overlap, or are on different chromosomes the value remains 0
    dists = np.zeros((len(exons)), dtype=np.int)

    ####################################
    # extraction of distances between exons and exon lengths

    # storing the END position of the current exon when overlapping,
    # avoids retrieving the END of the next one as a reference if smaller.
    prevEND = 0

    for exon in range(len(exons) - 1):
        # parameters to be compared
        currentCHR = exons[exon][0]
        currentSTART = exons[exon][1]
        currentEND = exons[exon][2]

        nextCHR = exons[exon + 1][0]
        nextSTART = exons[exon + 1][1]

        # extracts exon length
        lenEx.append(currentEND - currentSTART)

        # case exons on different chromosomes => next
        if (currentCHR != nextCHR):
            # must be reset at each chromosome change
            prevEND = 0 
            continue

        # the next overlapping exon is shorter than the previous one
        if prevEND != 0:
            # the old exon is longer than the current, keep the old END position
            if prevEND > currentEND:
                currentEND = prevEND

        # where the exons overlap no distance can be calculated => next
        if (nextSTART - currentEND) <= 0:
            prevEND = currentEND
            continue

        # reset because the exons don't overlap 
        prevEND = 0

        # extracts the distance between exons once the filters have been passed,
        # if not it remains at 0
        dists[exon] = (nextSTART - currentEND)

    # to identify the optimal distance between exons the median and mean
    # are too influenced by the small intronic distances.
    # we will only consider the bottom fracDistance fraction of distances, default
    # value should be fine
    fracDistance = 0.80
    # corresponding max distance value, don't take the distances at 0
    # which are the result of the different filtering
    interDist = np.int(np.quantile(dists[dists > 0], fracDistance))

    # identification of a threshold limit associated with the largest permissible inter-distance,
    # we assume that the 100 largest distances contain the centromeric regions
    largeDist = 100
    sortDist = sorted(dists, reverse=True)
    maxDist = np.int(sortDist[largeDist - 1])

    medLenEx = np.int(np.median(lenEx))


    ####################################
    # populating exonsAndInterRegion
    # unique naming of each intergenic window created with a counter
    windowIndex = 1

    for exon in range(len(exons) - 1):
        # parameters to be compared
        currentCHR = exons[exon][0]
        currentEND = exons[exon][2]

        # Fill current exon
        exonsAndInterRegion.append(exons[exon])

        # 0 is associated with overlapping exons or exons on different chromosomes
        if dists[exon] == 0:
            continue

        # skip centromeric regions
        if dists[exon] > maxDist:
            continue

        # the distance is too small to create new windows
        if dists[exon] < ((2 * interDist) + medLenEx):
            continue

        ######
        # new windows creation
        # factor(rounded down to the nearest integer) associated with the window number
        # to be inserted in the distance between exons
        nbWindows = np.int((dists[exon] - interDist) // (interDist + medLenEx))
        # new optimal distance deduced from the factor
        optiInterDist = np.round((dists[exon] - (nbWindows * medLenEx)) / (nbWindows + 1))

        # genomic position to define the coordinates for each new window
        coord = currentEND
        for i in range(nbWindows):
            coord += optiInterDist
            # Fill the new intergenic window
            exonsAndInterRegion.append([currentCHR, np.int(coord), np.int(coord + medLenEx),
                                        "intergenic_window_" + str(windowIndex)])
            coord += medLenEx
            windowIndex += 1

    # addition of the last exon
    exonsAndInterRegion.append(exons[-1])

    return exonsAndInterRegion
