#############################################################
############ Modules/Bed.py
#############################################################
import sys
import os
import re
import gzip
import logging

# set up logger, using inherited config
logger = logging.getLogger(__name__)

#############################################################
################ Function
#############################################################
#Exon intervals file parsing and preparing.
# Input : bedFile == a bed file (with path), possibly gzipped, containing exon definitions
#         formatted as CHR START END EXON_ID
#
# Output : returns a list of [numberOfExons] lists of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID, and where:
# - a padding is added to exon coordinates (ie -padding for START and +padding for END,
#   current padding=10bp)
# - exons are sorted by CHR, then START, then END, then EXON_ID
def processBed(bedFile):
    # number of bps used to pad the exon coordinates
    padding=10

    # list of exons to be returned
    exons=[]
    
    # we'll need a numerical version of CHR but we must read the whole file
    # beforehand => fill a CHR -> CHR_NUM dictionary
    translateCHRDict={}
    # temp dict containing all non-numeric chrs, key is the CHR stripped of 'chr' if
    # present and value is the CHR (e.g 'Y'->'chrY')
    remainingCHRs={}
    #maxCHR: max int value in CHR column (after stripping 'chr' if present)
    maxCHR=0

    #dictionary checking that the EXON_ID are unique(key='EXON_ID', value=1) 
    exonIDDict={}

    bedname=os.path.basename(bedFile)
    try:
        if bedname.endswith(".gz"):
            bedFH = gzip.open(bedFile, "rt")
        else:
            bedFH = open(bedFile, "r")
    except Exception as e:
        logger.error("Cannot open BED file - %s", e)
        raise Exception()

    for line in bedFH:
        fields = line.rstrip().split("\t")
        #############################
        # sanity checks + preprocess data
        # need exactly 4 fields
        if len(fields) != 4 :
            logger.error("In BED file %s, line doesn't have 4 fields:\n%s",
                         bedname, line)
            raise Exception()
        # START and END must be ints
        if fields[1].isdigit() and fields[2].isdigit():
            # OK, convert to actual ints and pad (but don't pad to negatives)
            fields[1] = max(int(fields[1]) - padding, 0)
            fields[2] = int(fields[2]) + padding
        else:
            logger.error("In BED file %s, columns 2-3 START-END are not ints in line:\n%s",
                         bedname, line)
            raise Exception()
        # EXON_ID must be unique
        if fields[3] in exonIDDict:
            logger.error("In BED file %s, EXON_ID (4th column) %s is not unique",
                         bedname, fields[3])
            raise Exception()
        #############################
        # prepare numeric version of CHR
        # we want ints so we remove chr prefix from CHR column if present
        chrNum = re.sub("^chr", "", fields[0])
        if chrNum.isdigit():
            chrNum = int(chrNum)
            translateCHRDict[fields[0]] = chrNum
            if maxCHR<chrNum:
                maxCHR=chrNum
        else:
            #non-numeric chromosome: save in remainingChRs for later
            remainingCHRs[chrNum]=fields[0]
        #############################
        # save exon definition
        exons.append(fields)

    ###############
    #### Non-numerical chromosome conversion to int
    ###############
    #replace non-numerical chromosomes by maxCHR+1, maxCHR+2 etc
    increment=1
    # first deal with X, Y, M/MT in that order
    for chrom in ["X","Y","M","MT"]:
        if chrom in remainingCHRs:
            translateCHRDict[remainingCHRs[chrom]] = maxCHR+increment
            increment+=1
            del remainingCHRs[chrom]
    # now deal with any other non-numeric CHRs, in alphabetical order
    for chrom in sorted(remainingCHRs):
        translateCHRDict[remainingCHRs[chrom]] = maxCHR+increment
        increment+=1
        
    ############### 
    #### finally we can add the CHR_NUM column to exons
    ###############
    for line in range(len(exons)):
        exons[line].append(translateCHRDict[exons[line][0]])

    ############### 
    #### Sort and remove temp CHR_NUM column
    ###############    
    # sort by CHR_NUM, then START, then END, then EXON_ID
    exons.sort(key = lambda row: (row[4],row[1],row[2],row[3]))
    # delete the tmp column, and return result
    for line in range(len(exons)):
        exons[line].pop()
    return(exons)
