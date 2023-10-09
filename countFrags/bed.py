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
# exonOnSexChr: identify exons located on a sex chromosome, ie currently one of
# (X, Y, Z, W). This covers most species including mammals, birds, fish, reptiles.
#
# Arg:
#  - a list of exons, each exon is a list of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID
#
# Returns an uint8 numpy.ndarray of the same size as exons, value is:
# - 0 if the exon is on an autosome;
# - 1 if the exon is on the X or Z chromosome;
# - 2 if the exon is on the Y or W chromosome.
#
# Note that X or Z is present in one or two copies in each individual, and is
# (usually?) the larger of the two sex chromosomes; while Y or W is present
# in 0 or 1 copy and is smaller.
# However interpretation of "having two copies of X|Z" differs: in XY species
# (eg humans) XX is the Female, while in ZW species (eg birds) ZZ is the Male.
def exonOnSexChr(exons):
    # accepted sex chromosomes as keys, value==1 for X|Z and 2 for Y|W
    sexChroms = {"X": 1, "Y": 2, "W": 2, "Z": 1}
    # also accept the same chroms prepended with 'chr'
    for sc in list(sexChroms.keys()):
        sexChroms["chr" + sc] = sexChroms[sc]

    exonOnSexChr = np.zeros(len(exons), dtype=np.uint8)
    for ei in range(len(exons)):
        if exons[ei][0] in sexChroms:
            exonOnSexChr[ei] = sexChroms[exons[ei][0]]
    return(exonOnSexChr)


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
#   of inter-exon distances (q == interExonQuantile hard-coded below)
# - if several pseudo-exons can fit in an inter-exon gap, produce as many as possible,
#   evenly spaced
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
    # - interExonQuantile-th quantile of inter-exon distances (selectedIED)
    # - longest inter-exon distance per chromosome
    medianExonLength = 0
    selectedIED = 0
    # dict: key==chrom, value==longest inter-exon distance
    chom2longestIED = {}

    # need to store the exon lengths and inter-exon distances, dealing
    # with overlapping exons
    exonLengths = []
    interExonDistances = []

    prevChrom = ""
    prevEnd = 0
    for exon in exons:
        # always store the exon length
        exonLengths.append(exon[2] - exon[1] + 1)
        if prevChrom != exon[0]:
            prevChrom = exon[0]
            prevEnd = exon[2]
            chom2longestIED[prevChrom] = 0
        else:
            # same chrom: store inter-exon dist if exons don't overlap
            interExDist = exon[1] - prevEnd - 1
            if interExDist > 0:
                interExonDistances.append(interExDist)
            if interExDist > chom2longestIED[prevChrom]:
                chom2longestIED[prevChrom] = interExDist
            # update prevEnd unless exon is fully included in prev
            prevEnd = max(prevEnd, exon[2])

    medianExonLength = int(np.median(exonLengths))
    selectedIED = int(np.quantile(interExonDistances, interExonQuantile))
    logger.info("Creating intergenic pseudo-exons: length = %i , interExonDistance = %i",
                medianExonLength, selectedIED)

    ############################
    # second pass: populate genomicWindows
    prevChrom = ""
    prevEnd = 0
    pseudoExonNum = 1
    for exon in exons:
        if prevChrom != exon[0]:
            prevChrom = exon[0]
            prevEnd = exon[2]
            genomicWindows.append(exon)
        else:
            interExDist = exon[1] - prevEnd - 1
            # number of pseudo-exons to create between previous exon and this exon
            # (negative if they overlap or are very close)
            pseudoExonCount = (interExDist - selectedIED) // (selectedIED + medianExonLength)
            if (interExDist < chom2longestIED[prevChrom]) and (pseudoExonCount > 0):
                # distance between (pseudo-)exons to use so they are evenly spaced
                thisIED = (interExDist - (pseudoExonCount * medianExonLength)) // (pseudoExonCount + 1)
                thisStart = prevEnd + thisIED + 1
                for i in range(pseudoExonCount):
                    # if EXON_ID for the intergenic pseudo-exons changes here, it MUST ALSO
                    # change in parseAndNormalizeCounts() in countsFile.py
                    genomicWindows.append([prevChrom, thisStart, thisStart + medianExonLength - 1,
                                           "intergenic_" + str(pseudoExonNum)])
                    # += (medianExonLength -1) + (thisIED + 1) , simplified to:
                    thisStart += medianExonLength + thisIED
                    pseudoExonNum += 1
            # whether pseudo-exons were created or not, update prevEnd and append exon
            prevEnd = max(prevEnd, exon[2])
            genomicWindows.append(exon)
    return(genomicWindows)
