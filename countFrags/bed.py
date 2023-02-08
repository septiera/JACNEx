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

    # for sorting we'll need a numerical version of CHR, but we must read the whole file
    # beforehand => fill a CHR -> CHR_NUM dictionary
    chr2num = {}
    # dict to store all non-numeric chromosomes, key is the CHR stripped of 'chr' if
    # present and value is the CHR (e.g 'Y'->'chrY')
    nonNumChrs = {}
    # maxCHR: max int value in CHR column (after stripping 'chr' if present)
    maxCHR = 0

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
        #############################
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
        #############################
        # populate chr2num with numeric version of CHR (if not seen yet)
        if fields[0] not in chr2num:
            # strip chr prefix from CHR if present
            if fields[0].startswith('chr'):
                chrStripped = fields[0][3:]
            else:
                chrStripped = fields[0]

            if chrStripped.isdigit():
                # numeric chrom: populate chr2num
                chrNum = int(chrStripped)
                chr2num[fields[0]] = chrNum
                if maxCHR < chrNum:
                    maxCHR = chrNum
            else:
                # non-numeric chromosome: save in nonNumChrs for later
                nonNumChrs[chrStripped] = fields[0]
                # also populate chr2num with dummy value, to skip this chr next time we see it
                chr2num[fields[0]] = -1
        #############################
        # save exon definition
        exons.append(fields)
    bedFH.close()

    #########################
    #### Done parsing bedFile
    #########################
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
    # add temp CHR_NUM column to exons
    for line in range(len(exons)):
        exons[line].append(chr2num[exons[line][0]])
    # sort exons by CHR_NUM, then START, then END, then EXON_ID
    exons.sort(key=lambda row: (row[4], row[1], row[2], row[3]))
    # delete the tmp column, and return result
    for line in range(len(exons)):
        exons[line].pop()
    return(exons)
