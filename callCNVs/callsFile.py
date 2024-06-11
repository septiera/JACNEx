############################################################################################
# Copyright (C) Nicolas Thierry-Mieg and Amandine Septier, 2021-2024
#
# This file is part of JACNEx, written by Nicolas Thierry-Mieg and Amandine Septier
# (CNRS, France)  {Nicolas.Thierry-Mieg,Amandine.Septier}@univ-grenoble-alpes.fr
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
############################################################################################


import gzip
import logging
import math
import numpy
import time

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
##########################################
# printCallsFile:
# print CNVsfor a single cluster in VCF format.
#
# Args:
# - outFile (str): name of VCF file to create. Will be squashed if pre-exists,
#   can include a path (which must exist), will be gzipped if outFile ends with '.gz'.
# - CNVs: list of CNVs, a CNV is a list (types [int, int, int, float, int]):
#   [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleIndex]
#   where firstExonIndex and lastExonIndex are indexes in the provided exons,
#   and sampleIndex is the index in the provided samples
# - FPMs: 2D-array of floats, size = nbExons * nbSamples, FPMs[e,s] is the FPM
#   count for exon e in sample s
# - CN2means: 1D-array of nbExons floats, CN2means[e] is the fitted mean of
#   the CN2 model of exon e for the cluster, or 0 if exon is NOCALL
# - samples: list of nbSamples sampleIDs (==strings)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - padding (int): padding bases used
# - madeBy (str): Name + version of program that made the CNV calls
# - refVcfFile (str): name (with path) of the VCF file for the reference cluster (ie cluster
#   without any FITWITHs) that serves as FITWITH for the current cluster, if any
# - clusterID (str): id of current cluster (for logging)
def printCallsFile(outFile, CNVs, FPMs, CN2Means, samples, exons, padding, madeBy, refVcfFile, clusterID):
    if refVcfFile != "":
        maxCalls = countCallsFromVCF(refVcfFile)
        CNVs = recalibrateGQs(CNVs, len(samples), maxCalls, clusterID)

    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open CNCallsFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open CNCallsFile')

    # Header
    toPrint = """##fileformat=VCFv4.3
##fileDate=""" + time.strftime("%y%m%d") + """
##source=""" + madeBy + """
##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=DUP,Description="Duplication">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Float,Description="Genotype quality score">
##FORMAT=<ID=FR,Number=1,Type=Float,Description="Fragment count ratio">
"""
    colNames = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
    toPrint += "\t".join(colNames) + "\n"
    outFH.write(toPrint)

    # CNVs: sort by chrom-start-end (ie by exonIndexes), and at identical coords we will
    # create (up to) 2 VCF lines: for DELs then DUPs -> sort by CNVType
    CNVs.sort(key=lambda CNV: (CNV[1], CNV[2], CNV[0], CNV[4]))

    prevVcfStart = []
    vcfGenos = []
    for CNV in CNVs:
        (cn, startExi, endExi, qualScore, sampleIndex) = CNV
        chrom = exons[startExi][0]
        pos = exons[startExi][1]
        if (pos == 0):
            logger.warning("exon %s START <= padding, cannot remove padding -> using POS=0 in VCF",
                           exons[startExi][3])
        else:
            pos += padding
        end = exons[endExi][2] - padding

        if cn <= 1:
            svtype = "DEL"
        else:
            svtype = "DUP"
        alt = '<' + svtype + '>'

        vcfStart = [chrom, str(pos), ".", ".", alt, ".", ".", f"SVTYPE={svtype};END={end}", "GT:GQ:FR"]

        if vcfStart != prevVcfStart:
            if len(prevVcfStart) > 0:
                toPrint = "\t".join(prevVcfStart + vcfGenos) + "\n"
                outFH.write(toPrint)
            prevVcfStart = vcfStart
            # default to all-HomoRef
            vcfGenos = ["0/0"] * len(samples)

        # calculate FragRatio: average of FPM ratios over all called exons in the CNV
        fragRat = 0
        numExons = 0
        for ei in range(startExi, endExi + 1):
            if CN2Means[ei] > 0:
                fragRat += FPMs[ei, sampleIndex] / CN2Means[ei]
                numExons += 1
            # else exon ei is NOCALL, ignore it
        fragRat /= numExons

        # genotype, depending on fragRat for DUPs: at a diploid locus we
        # expect fragRat ~1.5 for HET-DUPs and starting at ~2 for CN=4+ (which we
        # will call HV-DUPs, although maybe CN=3 on one allele and CN=1 on the other)...
        # [NOTE: at a haploid locus we expect ~2 for a hemizygous DUP but we're OK
        # to call those HV-DUP]
        # => hard-coded arbitrary cutoff between 1.5 and 2, closer to 2 to be conservative
        minFragRatDupHomo = 1.9
        geno = "1/1"
        if (cn == 1):
            geno = "0/1"
        elif (cn == 3) and (fragRat < minFragRatDupHomo):
            geno = "0/1"

        vcfGenos[sampleIndex] = f"{geno}:{qualScore:.1f}:{fragRat:.2f}"

    # print last CNV
    if len(prevVcfStart) > 0:
        toPrint = "\t".join(prevVcfStart + vcfGenos) + "\n"
        outFH.write(toPrint)
    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

##########################################
# countCallsFromVCF:
# count the number of calls per sample of each CN type, in vcfFile.
# Based on this, calculate a max expected number of calls per sample
# of each CN type for "similar" samples: the vcfFile holds calls made for
# a "reference" cluster, and the stats will be used to recalibrate the GQ
# scores of an associated cluster that uses this ref cluster as FITWITH,
# in order to obtain (at most) similar numbers of calls of each type.
#
# Args:
# - vcfFile: string, name of an existing VCF file (with path)
#
# Returns: maxCalls: list of 4 ints, maxCalls[CN] is the max number of allowed CN calls
# per sample (maxCalls[2] is ignored).
def countCallsFromVCF(vcfFile):
    try:
        if vcfFile.endswith(".gz"):
            vcfFH = gzip.open(vcfFile, "rt")
        else:
            vcfFH = open(vcfFile, "r")
    except Exception as e:
        logger.error("Cannot open VCF file - %s", e)
        raise Exception("countCallsFromVCF failed, check log")

    for line in vcfFH:
        if line.startswith('#CHROM'):
            numSamples = len(line.rstrip().split("\t")) - 9
            CN0perSample = numpy.zeros(numSamples, dtype=numpy.uint32)
            CN1perSample = numpy.zeros(numSamples, dtype=numpy.uint32)
            CN3perSample = numpy.zeros(numSamples, dtype=numpy.uint32)
        elif not line.startswith('#'):
            calls = line.rstrip().split("\t")
            svtype = calls[4]
            del calls[:9]
            for si in range(len(calls)):
                if calls[si].startswith('0/1:'):
                    if svtype == '<DEL>':
                        CN1perSample[si] += 1
                    else:
                        CN3perSample[si] += 1
                elif calls[si].startswith('1/1'):
                    if svtype == '<DEL>':
                        CN0perSample[si] += 1
                    else:
                        CN3perSample[si] += 1
    vcfFH.close()

    # accept up to hard-coded 80%-quantile of numbers of calls from the ref cluster
    numCallsQuantile = 0.8
    maxCalls = [0, 0, 0, 0]
    maxCalls[0] = numpy.quantile(CN0perSample, numCallsQuantile)
    maxCalls[1] = numpy.quantile(CN1perSample, numCallsQuantile)
    maxCalls[3] = numpy.quantile(CN3perSample, numCallsQuantile)
    return(maxCalls)


##########################################
# recalibrateGQs:
# for each CN type, calculate the minGQ corresponding to the provided max numbers
# of calls (on average per sample), and recalibrate the GQs accordingly (-= minGQ).
# Returns the recalibrated CNVs
def recalibrateGQs(CNVs, numSamples, maxCalls, clusterID):
    recalCNVs = []
    # GQs[CN] is the list of GQs of calls of type CN
    GQs = [[], [], [], []]
    for cnv in CNVs:
        GQs[cnv[0]].append(cnv[3])

    # minGQ[CN] is the min GQ that results in accepting at most maxCalls[CN] calls per
    # sample on average
    minGQ = [0, 0, 0, 0]
    for cn in (0, 1, 3):
        numAcceptedCalls = math.floor(numSamples * maxCalls[cn])
        if numAcceptedCalls < len(GQs[cn]):
            sortedGQs = sorted(GQs[cn], reverse=True)
            minGQ[cn] = sortedGQs[numAcceptedCalls]

    for cnv in CNVs:
        recalGQ = cnv[3] - minGQ[cnv[0]]
        if recalGQ >= 0:
            recalCNVs.append([cnv[0], cnv[1], cnv[2], recalGQ, cnv[4]])

    if minGQ != [0, 0, 0, 0]:
        logger.info("cluster %s - recalibrated GQs by -%.1f (CN0), -%.1f (CN1), -%.1f (CN3+)",
                    clusterID, minGQ[0], minGQ[1], minGQ[3])

    return(recalCNVs)
