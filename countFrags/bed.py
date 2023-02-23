import gzip
import logging

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
# - exons are sorted by CHR, then START, then END, then EXON_ID
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
    return(exons)


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
