import gzip
import logging
import time

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
##########################################
# printCallsFile
# Args :
# - vcf (list of lists)
# - outFile [str]: filename that doesn't exist, it can have a path component (which must exist),
#                  output will be gzipped if outFile ends with '.gz'
# - scriptName [str]
# - samples (list[strs]): sample names.
def printCallsFile(CNVs_A, CNVs_G, qs_A, qs_G, autosomeExons, gonosomeExons, samples, padding, outFile, scriptName):

    vcf_A = vcfFormat(CNVs_A, qs_A, autosomeExons, samples, padding)
    vcf_G = vcfFormat(CNVs_G, qs_G, gonosomeExons, samples, padding)

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
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype (always 0/1 for duplications)">
##FORMAT=<ID=QS,Number=1,Type=Float,Description="Call quality score">"""
    toPrint += "\n"
    outFH.write(toPrint)

    colNames = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
    outFH.write('\t'.join(colNames))
    outFH.write('\n')

    #### fill results
    for cnvIndex in range(len(vcf_A)):
        toPrint = '\t'.join(str(x) for x in vcf_A[cnvIndex])
        toPrint += "\n"
        outFH.write(toPrint)

    for cnvIndex in range(len(vcf_G)):
        toPrint = '\t'.join(str(x) for x in vcf_G[cnvIndex])
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
# vcfFormat
# formats CNV data into a Variant Call Format (VCF), organizing CNVs by chromosome,
# position, and type, and handling VCF-specific fields.
#
# Args:
# - CNVs (list): CNV information, each CNV formatted as [CNType, startExonIndex, endExonIndex,
#    probability, sampleID].
# - QS (list[floats]): Quality scores corresponding to each CNV.
# - exons (list): Exon information, each exon formatted as [chromosome, start, end, exonID].
# - samples (list[strs])
# - padding [int]: Value used to adjust start and end positions of CNVs.
#
# Returns:
# vcf (list[strs]): Each string represents a line in a VCF file, formatted with CNV information.
def vcfFormat(CNVs, QS, exons, samples, padding):
    # Define the number of columns before sample information in a VCF file
    # ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]
    infoColumns = 9

    # Initialize output list for VCF formatted data
    vcf = []

    # Initialize dictionary to store VCF lines keyed by chrom, pos, end, and type
    cnv_dict = {}

    # Iterate over CNVs
    for cnvIndex in range(len(CNVs)):
        cnvList = CNVs[cnvIndex]
        # Unpack CNV list into variables
        cn, startExi, endExi, prob, sampID = cnvList
        # Retrieve quality score for current CNV
        cnvQS = QS[cnvIndex]

        # Get chromosome, start and end position from exons
        chrom = exons[startExi][0]

        # remove padding
        pos = exons[startExi][1] + padding
        end = exons[endExi][2] - padding

        # Define SVTYPE and ALT based on CN value
        svtype = "DEL" if cn != 3 else "DUP"
        alt = "<DEL>" if cn != 3 else "<DUP>"

        # Create a tuple key for current CNV
        currentCNV = (chrom, pos, end, svtype)

        # Calculate sample index in VCF line
        sampi = samples.index(sampID) + infoColumns

        # Check if CNV is already processed, otherwise create a new VCF line
        if currentCNV not in cnv_dict:
            # Format the VCF line
            vcfLine = [chrom, pos, ".", ".", alt, ".", ".", f"SVTYPE={svtype};END={end}", "GT:QS"]

            # Initialize genotype annotations for all samples
            format = ["0/0"] * len(samples)
            vcfLine += format

        else:
            # Retrieve existing VCF line from dictionary
            vcfLine = cnv_dict[currentCNV]

        # Determine the sample's genotype based on CN type and add quality score
        geno = "1/1" if cn == 0 else "0/1"
        vcfLine[sampi] = f"{geno}:{cnvQS:.2f}"

        # Update the VCF line in the dictionary
        cnv_dict[currentCNV] = vcfLine

    logger.debug("total CNVs: %i, total aggregated CNVs: %i", len(CNVs), len(cnv_dict))

    sorted_tuple_list = sorted(cnv_dict, key=chromSort)

    # Add all processed VCF lines to the output list
    for cnv in sorted_tuple_list:
        vcf.append(cnv_dict[cnv])

    return vcf


#################################################
# chromSort
# sorting chromosomes, handling both numerical and special chromosomes (X, Y, M)
# for logical sorting.
#
# Args:
# - chrom_tuple (tuple): A tuple representing a CNV, the first element being the
#    chromosome identifier (e.g., 'chr1', 'chrX').
#
# Returns a tuple: Sorting key for chromosomes, ensuring numerical chromosomes
# are sorted numerically and special chromosomes in a predefined order.
def chromSort(chrom_tuple):
    chrom = chrom_tuple[0]
    # Extract the chromosome part after "chr"
    chrom_num = chrom[3:]
    # Assign an order value based on the chromosome type
    if chrom_num.isdigit():
        # Return a tuple (0, number) for numeric chromosomes, followed by other tuple elements
        return (0, int(chrom_num)) + chrom_tuple[1:]
    # Return a tuple with a higher first value for non-numeric chromosomes
    return (1, {"X": 1, "Y": 2, "M": 3}[chrom_num]) + chrom_tuple[1:]
