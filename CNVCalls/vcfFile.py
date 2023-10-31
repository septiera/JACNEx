import logging
import numpy as np
import time
import gzip

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
##############
# CNV2Vcf
# Given a list of CNV information, exon details, sample names, and a padding value.
# It converts the CNV data into Variant Call Format (VCF) by generating VCF lines
# based on the CNV information.
# Sorts the CNVs based on their start position, end position, and type.
# For each CNV, it creates or updates VCF lines, adjusting genotype information
# for samples based on CNV types. 
#
# Args:
# - CNVList (list of lists[int, int, int, str]) : an array of CNV informations [CN,ExonSTART,ExonEND,SAMPID]
# - exons (list of lists[str, int, int, str]): A list of exon informations [CHR,START,END,EXONID]
# - samples (list[strs]): sample names
# - padding [int]: user defined parameters (used in s1_countFrags.py)
#
# Returns:
# - a list of lists (List[List[Any]]) representing the VCF output.
#   Each inner list contains the information for a single VCF line.
def CNV2Vcf(CNVList, exons, samples, padding):
    infoColumns = 9  # ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]
    # Sort the list of lists based on multiple columns in a specific order
    # filtering step:  START,END,CN
    sorted_list = sorted(CNVList, key=lambda x: (x[1], x[2], x[0]))

    vcf = []  # output list

    # Dictionary to store CNV information by key (chrom, pos, end, type)
    cnv_dict = {}

    for cnvIndex in range(len(sorted_list)):
        cnvInfo = sorted_list[cnvIndex]
        chrom = exons[cnvInfo[1]][0]  # str
        # remove padding
        pos = exons[cnvInfo[1]][1] + padding  # int
        end = exons[cnvInfo[2]][2] - padding  # int
        CNtype = cnvInfo[0]  # int
        currentCNV = (chrom, pos, end, CNtype)

        # CNV with the same key already exists in the dictionary,
        # update the corresponding sample's value in the VCF line
        if currentCNV in cnv_dict:
            vcfLine = cnv_dict[currentCNV]
            # Get the index of the sample in the VCF line
            # (+9 to account for the non-sample columns)
            sampleIndex = samples.index(cnvInfo[3])
            
            # Determine the sample's genotype based on CNV type
            sampleInfo = "1/1" if CNtype == 0 else "0/1"
            vcfLine[sampleIndex] = sampleInfo

        # CNV location and type not seen before, create a new VCF line
        else:
            if (CNtype != 3):
                vcfLine = [chrom, pos, ".", ".", "<DEL>", ".", ".", "SVTYPE=DEL;END=" + str(end), "GT"]
            else:
                vcfLine = [chrom, pos, ".", ".", "<DUP>", ".", ".", "SVTYPE=DUP;END=" + str(end), "GT"]
            # Set CN2 default values for all samples columns
            sampleInfo = ["0/0"] * len(samples)
            sampleIndex = samples.index(cnvInfo[3])
            print(sampleIndex,cnvIndex, currentCNV)
            sampleInfo[sampleIndex] = "1/1" if CNtype == 0 else "0/1"
            vcfLine += sampleInfo
            # Store the VCF line in the dictionary for future reference
            cnv_dict[currentCNV] = vcfLine
            vcf.append(vcfLine)

    return vcf


##########################################
# printVcf
# Args :
# - vcf (list of lists)
# - outFile [str]: filename that doesn't exist, it can have a path component (which must exist),
#                  output will be gzipped if outFile ends with '.gz'
# - scriptName [str]
# - samples (list[strs]): sample names.
def printVcf(vcf, outFile, scriptName, samples):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open CNCallsFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open CNCallsFile')

    # Header definition
    toPrint = """##fileformat=VCFv4.3
##fileDate=""" + time.strftime("%y%m%d") + """
##source=""" + scriptName + """
##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=DUP,Description="Duplication">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype (always 0/1 for duplications)">"""
    toPrint += "\n"
    outFH.write(toPrint)

    colNames = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
    outFH.write('\t'.join(colNames))

    #### fill results
    for cnvIndex in range(len(vcf)):
        toPrint = '\t'.join(str(x) for x in vcf[cnvIndex])
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()
