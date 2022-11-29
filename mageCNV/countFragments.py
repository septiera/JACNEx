import os
import numpy as np
import numba  # make python faster
import re
# nested containment lists, similar to interval trees but faster (https://github.com/biocore-ntnu/ncls)
from ncls import NCLS
import subprocess
import tempfile
import time
import logging

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# countFrags :
# Count the fragments in bamFile that overlap each exon described in exons.
# Arguments:
#   - a bam file (with path)
#   - exon definitions as returned by processBed, padded and sorted
#   - the maximum accepted gap length between paired reads
#   - a fast tmp dir with enough space for samtools collate
#   - the samtools binary, with path
#   - the number of cpu threads that samtools can use
#
# Return a 1D numpy int array, dim = len(exons), with the fragment counts for this sample
def countFrags(bamFile, exons, maxGap, tmpDir, samtools, samThreads):
    startTime = time.time()
    # for each chrom, build an NCL holding the exons
    # NOTE: we would like to build this once in the caller and use it for each BAM,
    # but the multiprocessing module doesn't allow this... We therefore rebuild the
    # NCLs for each BAM, wasteful but it's OK, createExonNCLs() is fast
    exonNCLs = createExonNCLs(exons)

    thisTime = time.time()
    logger.debug("Done createExonNCLs for %s, in %.2f s",os.path.basename(bamFile), thisTime-startTime)
    startTime = thisTime

    # We need to process all alignments for a given qname simultaneously
    # => ALGORITHM:
    # parse alignements from BAM, grouped by qname;
    # if qname didn't change -> just apply some QC and if AOK store the
    #   data we need in accumulators,
    # if the qname changed -> process accumulated data for the previous
    #   qname (with Qname2ExonCount), then reset accumulators and store
    #   data for new qname.

    # Accumulators: 
    # QNAME and CHR
    qname, qchrom = "", ""
    # START and END coordinates on the genome of each alignment for this qname,
    # aligning on the Forward or Reverse genomic strands
    qstartF, qstartR, qendF, qendR = [], [], [], []
    # coordinate of the leftmost non-clipped base on the read, alis in the same order
    # as in qstartF/R and qendF/R
    qstartOnReadF, qstartOnReadR = [], []
    # qFirstOnForward==1 if the first-in-pair read of this qname is on the
    # forward reference strand, -1 if it's on the reverse strand, 0 if we don't yet know
    qFirstOnForward=0
    # qBad==True if qname must be skipped (e.g. alis on bad or multiple chroms, or alis
    # disagree regarding the strand on which the first/last read-in-pair aligns, or...)
    qBad = False

    # To Fill:
    # 1D numpy array containing the sample fragment counts for all exons
    sampleCounts=np.zeros(len(exons), dtype=np.uint32)

    ############################################
    # Preprocessing:
    # - need to parse the alignements grouped by qname, "samtools collate" allows this;
    # - we can also immediately filter out poorly mapped (low MAPQ) or dubious/bad
    #   alignments based on the SAM flags
    # Requiring:
    #   1 0x1 read paired
    # Discarding when any if the below is set:
    #   4 0x4 read unmapped
    #   8 0x8 mate unmapped
    #   256 0x80 not primary alignment
    #   512 0x200 read fails platform/vendor quality checks
    #   1024 0x400 read is PCR or optical duplicate
    #   -> sum == 1804
    # For more details on FLAGs read the SAM spec: http://samtools.github.io/hts-specs/
    tmpDirObj = tempfile.TemporaryDirectory(dir=tmpDir)
    tmpDirPrefix = tmpDirObj.name + "/tmpcoll"

    cmd = [samtools, 'collate', '-O', '--output-fmt', 'SAM', '--threads', str(samThreads)]
    cmd.extend(['--input-fmt-option', 'filter=(mapq >= 20) && flag.paired && !(flag & 1804)'])
    cmd.extend([bamFile, tmpDirPrefix])
    samproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

    ############################################
    # Main loop: parse each alignment
    for line in samproc.stdout:
        # skip header
        if re.match('^@', line):
            continue

        align=line.split('\t', maxsplit=6)

        ######################################################################
        # If we are done with previous qname: process it and reset accumulators
        if (qname!=align[0]) and (qname!=""):  # align[0] is the qname
            if not qBad:
                # if we have 2 alis on a strand, make sure they are in "read" order (switch them if needed)
                if (len(qstartF)==2) and (qstartOnReadF[0] > qstartOnReadF[1]):
                    qstartF.reverse()
                    qendF.reverse()
                if (len(qstartR)==2) and (qstartOnReadR[0] > qstartOnReadR[1]):
                    qstartR.reverse()
                    qendR.reverse()
                Qname2ExonCount(qstartF,qendF,qstartR,qendR,exonNCLs[qchrom],sampleCounts,maxGap)
            qname, qchrom = "", ""
            qstartF, qstartR, qendF, qendR = [], [], [], []
            qstartOnReadF, qstartOnReadR = [], []
            qFirstOnForward=0
            qBad=False
        elif qBad: #same qname as previous ali but we know it's bad -> skip
            continue

        ######################################################################
        # Either we're in the same qname and it's not bad, or we changed qname
        # -> in both cases update accumulators with current line
        if qname=="" :
            qname=align[0]

        # align[2] == chrom
        if align[2] not in exonNCLs:
            # ignore qname if ali not on a chromosome where we have at least one exon;
            # importantly this skips non-main GRCh38 "chroms" (eg ALT contigs)
            qBad=True
            continue
        elif qchrom=="":
            qchrom=align[2]
        elif qchrom!=align[2]:
            # qname has alignments on different chroms, ignore it
            qBad=True
            continue
        # else same chrom, keep going

        #Retrieving flags for STRAND and first/second read
        currentFlag=int(align[1])

        # currentFirstOnForward==1 if according to this ali, the first-in-pair
        # read is on the forward strand, -1 otherwise
        currentFirstOnForward=-1
        # flag 16 the alignment is on the reverse strand
        # flag 64 the alignment is first in pair, otherwise 128
        if ((currentFlag&80) == 64) or ((currentFlag&144) == 144) :
            currentFirstOnForward=1
        if qFirstOnForward==0:
            # first ali for this qname
            qFirstOnForward = currentFirstOnForward
        elif qFirstOnForward != currentFirstOnForward :
            qBad=True
            continue
        # else this ali agrees with previous alis for qname -> NOOP

        # START and END coordinates of current alignment on REF (ignoring any clipped bases)
        currentStart=int(align[3])
        # END depends on CIGAR
        currentCigar=align[5]
        currentAliLength=aliLengthOnRef(currentCigar)
        currentEnd=currentStart+currentAliLength-1
        # coord of leftmost non-clipped base on read
        currentStartOnRead = firstNonClipped(currentCigar)
        if (currentFlag&16) == 0:
            # flag 16 off => forward strand
            qstartF.append(currentStart)
            qendF.append(currentEnd)
            qstartOnReadF.append(currentStartOnRead)
        else:
            qstartR.append(currentStart)
            qendR.append(currentEnd)
            qstartOnReadR.append(currentStartOnRead)

    #################################################################################################
    # Done reading lines from BAM

    # process last Qname
    if (qname!="") and not qBad:
        # if we have 2 alis on a strand, make sure they are in "read" order (switch them if needed)
        if (len(qstartF)==2) and (qstartOnReadF[0] > qstartOnReadF[1]):
            qstartF.reverse()
            qendF.reverse()
        if (len(qstartR)==2) and (qstartOnReadR[0] > qstartOnReadR[1]):
            qstartR.reverse()
            qendR.reverse()
        Qname2ExonCount(qstartF,qendF,qstartR,qendR,exonNCLs[qchrom],sampleCounts,maxGap)

    # wait for samtools to finish cleanly and check exit code
    if (samproc.wait() != 0):
        logger.error("in countFrags, while processing %s, samtools exited with code %s",
                     bamFile, samproc.returncode)
        raise Exception("samtools failed")

    # tmpDirObj should get cleaned up automatically but sometimes samtools tempfiles
    # are in the process of being deleted => sync to avoid race
    os.sync()
    thisTime = time.time()
    logger.debug("Done countFrags for %s, in %.2f s",os.path.basename(bamFile), thisTime-startTime)
    return(sampleCounts)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# aliLengthOnRef :
#Input : a CIGAR string
#Output : span of the alignment on the reference sequence, ie number of bases
# consumed by the alignment on the reference
def aliLengthOnRef(cigar):
    length=0
    # only count CIGAR operations that consume the reference sequence, see CIGAR definition 
    # in SAM spec available here: https://samtools.github.io/hts-specs/
    match = re.findall(r"(\d+)[MDN=X]",cigar)
    for op in match:
        length+=int(op)
    return(length)

####################################################
# firstNonClipped:
# Arg: a CIGAR string
# Returns the coordinate on the read of the leftmost non-clipped base
def firstNonClipped(cigar):
    firstNonClipped = 1
    # count all leading H and S bases
    while True:
        match = re.match(r"^(\d+)[HS]", cigar)
        if match:
            firstNonClipped += int(match.group(1))
            cigar = re.sub(r"^(\d+)[HS]", '', cigar)
        else:
            break
    return(firstNonClipped)

#############################################################
# Create nested containment lists (similar to interval trees but faster), one per
# chromosome, storing the exons
# Input : list of exons (ie lists of 4 fields), as returned by processBed
# Output: dictionary(hash): key=chr, value=NCL
def createExonNCLs(exons):
    # for each chrom, build 3 lists with same length: starts, ends, indexes (in
    # the complete exons list). key is the CHR
    starts = {}
    ends = {}
    indexes = {}
    for i in range(len(exons)):
        # exons[i] is a list: CHR, START, END, EXON_ID
        chrom = exons[i][0]
        if chrom not in starts:
            # first time we see chrom, initialize with empty lists
            starts[chrom] = []
            ends[chrom] = []
            indexes[chrom] = []
        # in all cases, append current values to the lists
        starts[chrom].append(exons[i][1])
        ends[chrom].append(exons[i][2])
        indexes[chrom].append(i)

    # build dictionary of NCLs, one per chromosome
    exonNCLs = {}
    for chrom in starts.keys():
        ncl = NCLS(starts[chrom], ends[chrom], indexes[chrom])
        exonNCLs[chrom]=ncl
    return(exonNCLs)

####################################################
# Qname2ExonCount :
# Given data representing all alignments for a single qname (must all map to the same chrom):
# - apply QC filters to ignore aberrant or unusual alignments / qnames;
# - identify the genomic intervals that are covered by the sequenced fragment;
# - identify exons overlapped by these intervals, and increment their count.
# Args:
#   - 4 lists of ints for F and R strand positions (start and end), in "read" order
#   - the NCL for the chromosome where current alignments map
#   - the counts vector to update (1D numpy int array)
#   - the maximum accepted gap length between a pair of mate reads, pairs separated by
#     a longer gap are ignored (putative structural variant or alignment error)
# Nothing is returned, this function just updates countsVec
def Qname2ExonCount(startFList,endFList,startRList,endRList,exonNCL,countsVec,maxGap):
    # skip Qnames that don't have alignments on both strands
    if (len(startFList)==0) or (len(startRList)==0):
        return
    # skip Qnames that have at least 3 alignments on a strand
    if (len(startFList)>2) or (len(startRList)>2):
        return

    # if we have 2 alis on one strand and they overlap: merge them
    # (there could be a short indel or other SV, not trying to detect
    # this but we still cover this genomic region with this qname)
    if (len(startFList)==2) and (min(endFList) > max(startFList)):
        startFList = [min(startFList)]
        endFList = [max(endFList)]
    if (len(startRList)==2) and (min(endRList) > max(startRList)):
        startRList = [min(startRList)]
        endRList = [max(endRList)]
    
    # if we have 2 alis on one strand (eg F) and one ali on the other (R),
    # discard the rightmost F ali if it is back-to-back with the R ali
    # 2F 1R
    if (len(startFList)==2) and (len(startRList)==1):
        if max(startFList) > endRList[0]:
            startFList = [min(startFList)]
            endFList = [min(endFList)]
    # 1F 2R
    if (len(startFList)==1) and (len(startRList)==2):
        if min(endRList) < startFList[0]:
            startRList = [max(startRList)]
            endRList = [max(endRList)]

    if (len(startFList)==1) and (len(startRList)==1): # 1F1R
        if startFList[0] > endRList[0]:
            # alignments are back-to-back (SV? Dup? alignment error?)
            return
    # elif (len(startFList+startRList)==3)
    # 1F2R and 2F1R have either become 1F1R, or they can't have any back-to-back pair
    elif (len(startFList)==2) and (len(startRList)==2): #2F2R
        if (min(startFList) > min(endRList)) or (min(endFList) < min(startRList)):
            # leftmost F and R alignments are back-to-back, or they don't
            # overlap - but what could explain this? aberrant
            return
        elif (max(startFList) > max(endRList)) or (max(endFList) < max(startRList)):
            # rightmost F and R alignments are back-to-back or don't overlap
            return

    # Examine gap length between the two reads (negative if overlapping).
    # maxGap should be set so that the sequencing library fragments are rarely
    # longer than maxGap+2*readLength, otherwise some informative qnames will be skipped
    if (len(startFList)==1) and (len(startRList)==1): # 1F1R
        if (startRList[0] - endFList[0] > maxGap):
            # large gap between forward and reverse reads, could be a DEL but
            # we don't have reads spanning it -> insufficient evidence, skip qname
            return
    elif (len(startFList+startRList)==3): # 2F1R or 1F2R
        if (min(startRList)-max(endFList) > maxGap):
            # eg 2F1R: the R ali is too far from the rightmost F ali, so
            # a fortiori it is too far from the leftmost F ali => ignore qname
            return
        elif (max(startRList)-min(endFList) > maxGap):
            # eg 2F1R: the R ali is too far from the leftmost F ali => ignore this F ali
            if (len(startFList)==2): #2F1R
                startFList = [max(startFList)]
                endFList = [max(endFList)]
            else: #1F2R
                startRList = [min(startRList)]
                endRList = [min(endRList)]
                 
    #######################################################
    # identify genomic regions covered by the fragment
    #######################################################
    # fragStarts, fragEnds: start and end coordinates of the genomic intervals
    # covered by this fragment.
    # There is usually one (1F1R overlapping) or 2 intervals (1F1R not 
    # overlapping) , but if the fragment spans a DEL there can be up to 3
    fragStarts = []
    fragEnds = []
    # algo: merge F-R pairs of alis only if they overlap
    if (len(startFList)==1) and (len(startRList)==1):
        if (startRList[0] - endFList[0] < 0): # overlap => merge
            fragStarts = [min(startFList + startRList)]
            fragEnds = [max(endFList + endRList)]
        else:
            fragStarts = [startFList[0], startRList[0]]
            fragEnds = [endFList[0], endRList[0]]

    elif (len(startFList)==2) and (len(startRList)==1): #2F1R
        if (startRList[0] < min(endFList)): 
            # leftmost F ali overlaps R ali => merge into a single interval (because we know
            # that the rightmost F ali overlaps the R ali
            fragStarts = [min(startRList + startFList)]
            fragEnds = [max(endRList + endFList)]
        else:
            # leftmost F ali is an interval by itself
            fragStarts = [min(startFList)]
            fragEnds = [min(endFList)]
            if (startRList[0] < max(endFList)):
                # rightmost F ali overlaps R ali => merge
                fragStarts.append(min(max(startFList),startRList[0]))
                fragEnds.append(max(endFList + endRList))
            else:
                # no overlap, add 2 more intervals
                fragStarts.extend([max(startFList), startRList[0]])
                fragEnds.extend([max(endFList), endRList[0]])

    elif (len(startFList)==1) and (len(startRList)==2): #1F2R
        if (max(startRList) < endFList[0]): 
            # rightmost R ali overlaps F ali => merge into a single interval
            fragStarts = [min(startRList + startFList)]
            fragEnds = [max(endRList + endFList)]
        else:
            # rightmost R ali is an interval by itself
            fragStarts = [max(startRList)]
            fragEnds = [max(endRList)]
            if (min(startRList) < endFList[0]):
                # leftmost R ali overlaps F ali => merge
                fragStarts.append(min(startFList + startRList))
                fragEnds.append(max(min(endRList),endFList[0]))
            else:
                # no overlap, add 2 more intervals
                fragStarts.extend([min(startRList), startFList[0]])
                fragEnds.extend([min(endRList), endFList[0]])

    elif(len(startFList)==2) and (len(startRList)==2): #2F2R
        # we already checked that each pair of F-R alis overlaps
        fragStarts.extend([min(startFList + startRList), min(max(startFList),max(startRList))])
        fragEnds.extend([max(min(endFList),min(endRList)), max(endFList + endRList)])

    #######################################################
    # find exons overlapped by the fragment and increment counters
    #######################################################
    # we want to increment countsVec at most once per exon, even if
    # we have two intervals and they both overlap the same exon
    exonsSeen = []
    for fi in range(len(fragStarts)):
        overlappedExons = exonNCL.find_overlap(fragStarts[fi],fragEnds[fi])
        for exon in overlappedExons:
            exonIndex = exon[2]
            if (exonIndex in exonsSeen):
                continue
            else:
                exonsSeen.append(exonIndex)
                incrementCount(countsVec, exonIndex)
                # countsVec[exonIndex] += 1
    
####################################################
# incrementCount:
# increment the counter in countsVec (1D numpy array) at index exonIndex
@numba.njit
def incrementCount(countsVec, exonIndex):
    countsVec[exonIndex] += 1
