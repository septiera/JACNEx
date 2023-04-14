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
# Returns a list of [numberOfExons + numberOfPseudoExons] lists of 4 scalars
# (types: str,int,int,str) containing CHR,START,END,EXON_ID, and where:
# - exons from bedFile are padded, ie -padding for START (never negative) and +padding for END
# - pseudo-exons are inserted between consecutive exons when they are far apart, as specified
#   by insertPseudoExons()
# - exons and pseudo-exons are sorted by CHR, then START, then END, then EXON_ID
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

    # Done parsing bedFile -> sort exons, insert pseudo-exons, and return
    sortExonsOrBPs(exons)
    genomicWindows = insertPseudoExons(exons)
    return(genomicWindows)


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


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# Given a list of exons, return a similar list containing the original exons as
# well as "pseudo-exons", which are inserted between each pair of consecutive
# exons provided that the exons are far enough apart. The goal is to produce
# pseudo-exons in intergenic (or very long intronic) regions.
# Specifically:
# - EXON_ID for pseudo-exons is 'intergenic_N' (N is an int)
# - pseudo-exon length = median of exon lengths
# - minimum distance between a pseudo-exon and an exon or pseudo-exon = the q-th quantile
#   of inter-exon distances (q == interExonQuantile defined below)
# - if several pseudo-exons can fit in an inter-exon gap, produce as many as possible
# - pseudo-exons are evenly spaced in their inter-exon gap
# - the longest inter-exon gap on each chromosome is not populated with pseudo-exons (goal:
#    avoid centromeres)
#
# Arg:
# - exons == list of lists [str, int, int, str] containing CHR,START,END,EXON_ID,
#   sorted by sortExonsOrBPs()
#
# Returns a new sorted list of exon-like structures, comprising the exons + pseudo-exons
def insertPseudoExons(exons):
    # to fill and return
    genomicWindows = []

    # interExonQuantile: hard-coded between 0 and 1, a larger value increases the distance
    # between pseudo-exons and exons (and therefore decreases the total number of pseudo-exons)
    interExonQuantile = 0.8

    ############################
    # first pass, determine:
    # - median exon length
    # - longest inter-exon distance per chromosome
    # - interExonQuantile-th quantile of inter-exon distances


    ############################
    # second pass: populate and return genomicWindows



    
    
####################################################
# addIntergenicWindows
# To generate a reliable non-coverage profile, intergenic regions can be added to canonical exons.
# The goal is to create intergenic windows that do not overlap with any canonical exon, with an
# optimal spacing between them.
# This can be achieved by maintaining a natural pattern, which involves calculating the median
# length of exons and determining the distance between exons based on the largest majority of exons
# (without overlap or chromosomal difference).
# Arg:
# - exons (list of list[str, int, int, str]): exons definition [CHR, START, END, EXON_ID]
# The output format remains sorted even after the addition of intergenic regions, with the fourth column
# indicating unique region names in the format 'intergenic_window_n'.
def addIntergenicWindows(exons):
    # To Fill and return
    genomicWindows = []

    # To Fill not returns
    # list of int containing the lengths of each exon
    lenEx = []

    # numpy array of distances(bp) between exons is set to zero if the exons overlap or are located
    # on different chromosomes. In these cases, the distance remains zero.
    dists = np.zeros((len(exons)), dtype=np.int)

    ####################################
    # extraction of exon lengths and distances between exons

    # to avoid retrieving the END position of the next exon as a reference if it is smaller,
    # the END position of the current exon is stored when there is overlap.
    prevEND = 0

    # an integer variable is used to store the indexes corresponding to the beginning of each chromosome.
    # this variable is then used to filter out the centromeric regions for each chromosome.
    startChrom = 0

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
        if (currentCHR != nextCHR) or (exon == len(exons) - 2):
            # must be reset at each chromosome change
            prevEND = 0

            # to avoid including centromeric regions
            max_index = np.argmax(dists[startChrom:exon])
            dists[startChrom:exon][max_index] = 0
            startChrom = exon + 1
            continue

        # the next overlapping exon is potentially shorter than the previous one
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

        # distance between exons is extracted only if it passes through the filters;
        # otherwise, it remains at 0.
        dists[exon] = (nextSTART - currentEND)

    # to determine the optimal distance between exons, we avoid relying solely on the median and mean,
    # which may be heavily influenced by small intronic distances.
    # instead, we only consider the bottom fraction of distances, with the default
    # value being sufficient.
    fracDistance = 0.80
    # corresponding maximum distance value should not include distances that have been filtered
    # out and set to 0.
    interDist = np.int(np.quantile(dists[dists > 0], fracDistance))

    medLenEx = np.int(np.median(lenEx))

    ####################################
    # populating genomicWindows
    # unique naming of each intergenic window created with a counter
    windowIndex = 1

    for exon in range(len(exons) - 1):
        # parameters to be compared
        currentCHR = exons[exon][0]
        currentEND = exons[exon][2]

        # Fill current exon
        genomicWindows.append(exons[exon])

        # 0 is associated with overlapping exons or exons on different chromosomes
        if dists[exon] == 0:
            continue

        # the distance is too small to create new windows
        if dists[exon] < ((2 * interDist) + medLenEx):
            continue

        ######
        # A new set of windows is created based on the factor rounded down to the nearest integer,
        # which is associated with the window number to be inserted between the distances of exons.
        nbWindows = np.int((dists[exon] - interDist) // (interDist + medLenEx))
        # new optimal distance deduced from the factor
        optiInterDist = np.round((dists[exon] - (nbWindows * medLenEx)) / (nbWindows + 1))

        # genomic position to define the coordinates for each new window
        coord = currentEND
        for i in range(nbWindows):
            coord += optiInterDist
            # Fill the new intergenic window
            genomicWindows.append([currentCHR, np.int(coord), np.int(coord + medLenEx),
                                   "intergenic_window_" + str(windowIndex)])
            coord += medLenEx
            windowIndex += 1

    # addition of the last exon
    genomicWindows.append(exons[-1])

    return genomicWindows
