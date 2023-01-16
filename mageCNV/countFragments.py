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
#   - an int (sampleIndex) that is not used but is simply returned (for multiprocessing)
#
# Return a 3-element tuple (sampleIndex, sampleCounts, breakPoints) where:
# - sampleCounts is a 1D numpy int array dim = len(exons) allocated here and filled with
#   the counts for this sample,
# - breakPoints is a list of 5-element lists [CHR, BP1, BP2, CNVTYPE, QNAME], where
#   BP1 and BP2 are the coordinates of the putative breakpoints, CNVTYPE is 'DEL' or 'DUP',
#   and QNAME is the supporting fragment
# If anything goes wrong, log info on exception and then always raise Exception(str(sampleIndex)),
# so caller can catch it and know which sampleIndex we were working on.
def countFrags(bamFile, exons, maxGap, tmpDir, samtools, samThreads, sampleIndex):
    try:
        logger.info('Processing BAM %s', os.path.basename(bamFile))
        startTime = time.time()
        # for each chrom, build an NCL holding the exons
        # NOTE: we would like to build this once in the caller and use it for each BAM,
        # but the multiprocessing module doesn't allow this... We therefore rebuild the
        # NCLs for each BAM, wasteful but it's OK, createExonNCLs() is fast
        exonNCLs = createExonNCLs(exons)

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
        qFirstOnForward = 0
        # qBad==True if qname must be skipped (e.g. alis on multiple chroms, or alis
        # disagree regarding the strand on which the first/last read-in-pair aligns, or...)
        qBad = False

        # To Fill:
        # 1D numpy array containing the sample fragment counts for all exons
        sampleCounts = np.zeros(len(exons), dtype=np.uint32)
        # list of lists with info about breakpoints support
        breakPoints = []

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

            align = line.split('\t', maxsplit=6)

            ######################################################################
            # If we are done with previous qname: process it and reset accumulators
            if (qname != align[0]) and (qname != ""):  # align[0] is the qname
                # qchrom == "" is possible if all alis for qname mapped to non-main chroms
                if (not qBad) and (qchrom != ""):
                    # if we have 2 alis on a strand, make sure they are in "read" order (switch them if needed)
                    if (len(qstartF) == 2) and (qstartOnReadF[0] > qstartOnReadF[1]):
                        qstartF.reverse()
                        qendF.reverse()
                    if (len(qstartR) == 2) and (qstartOnReadR[0] > qstartOnReadR[1]):
                        qstartR.reverse()
                        qendR.reverse()
                    BPs = Qname2ExonCount(qstartF, qendF, qstartR, qendR, exonNCLs[qchrom], sampleCounts, maxGap)
                    if (BPs is not None):
                        BPs.insert(0, qchrom)
                        BPs.append(qname)
                        breakPoints.append(BPs)
                qname, qchrom = "", ""
                qstartF, qstartR, qendF, qendR = [], [], [], []
                qstartOnReadF, qstartOnReadR = [], []
                qFirstOnForward = 0
                qBad = False
            elif qBad:  # same qname as previous ali but we know it's bad -> skip
                continue

            ######################################################################
            # Either we're in the same qname and it's not bad, or we changed qname
            # -> in both cases update accumulators with current line
            if qname == "":
                qname = align[0]

            # align[2] == chrom
            if align[2] not in exonNCLs:
                # ignore alis that don't map to a chromosome where we have at least one exon;
                # importantly this skips alis on non-main GRCh38 "chroms" (eg ALT contigs)
                # Note: we ignore the ali but don't set qBad, because some tools report alignments to
                # ALT contigs in the XA tag ("secondary alignemnt") of the main chrom ali (as expected),
                # but also produce the alignements to ALTs as full alignement records with flag 0x800
                # "supplementary alignment" set, this is contradictory but we have to deal with it...
                continue
            elif qchrom == "":
                qchrom = align[2]
            elif qchrom != align[2]:
                # qname has alignments on different chroms, ignore it
                qBad = True
                continue
            # else same chrom, keep going

            # Retrieving flags for STRAND and first/second read
            currentFlag = int(align[1])

            # currentFirstOnForward==1 if according to this ali, the first-in-pair
            # read is on the forward strand, -1 otherwise
            currentFirstOnForward = -1
            # flag 16 the alignment is on the reverse strand
            # flag 64 the alignment is first in pair, otherwise 128
            if ((currentFlag & 80) == 64) or ((currentFlag & 144) == 144):
                currentFirstOnForward = 1
            if qFirstOnForward == 0:
                # first ali for this qname
                qFirstOnForward = currentFirstOnForward
            elif qFirstOnForward != currentFirstOnForward:
                qBad = True
                continue
            # else this ali agrees with previous alis for qname -> NOOP

            # START and END coordinates of current alignment on REF (ignoring any clipped bases)
            currentStart = int(align[3])
            # END depends on CIGAR
            currentCigar = align[5]
            currentAliLength = aliLengthOnRef(currentCigar)
            currentEnd = currentStart + currentAliLength - 1
            # coord of leftmost non-clipped base on read
            currentStartOnRead = firstNonClipped(currentCigar)
            if (currentFlag & 16) == 0:
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
        if (qname != "") and not qBad:
            # if we have 2 alis on a strand, make sure they are in "read" order (switch them if needed)
            if (len(qstartF) == 2) and (qstartOnReadF[0] > qstartOnReadF[1]):
                qstartF.reverse()
                qendF.reverse()
            if (len(qstartR) == 2) and (qstartOnReadR[0] > qstartOnReadR[1]):
                qstartR.reverse()
                qendR.reverse()
            BPs = Qname2ExonCount(qstartF, qendF, qstartR, qendR, exonNCLs[qchrom], sampleCounts, maxGap)
            if (BPs is not None):
                BPs.insert(0, qchrom)
                BPs.append(qname)
                breakPoints.append(BPs)

        # wait for samtools to finish cleanly and check exit code
        if (samproc.wait() != 0):
            logger.error("in countFrags, while processing %s, samtools exited with code %s",
                         bamFile, samproc.returncode)
            raise Exception("samtools failed")

        # tmpDirObj should get cleaned up automatically but sometimes samtools tempfiles
        # are in the process of being deleted => sync to avoid race
        os.sync()
        # we want breakpoints sorted by chrom (but not caring that chr10 comes before chr2),
        # then BP1 then BP2 then CNVTYPE
        breakPoints.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        thisTime = time.time()
        logger.debug("Done countFrags for %s, in %.2f s", os.path.basename(bamFile), thisTime - startTime)
        return(sampleIndex, sampleCounts, breakPoints)

    except Exception as e:
        logger.error(repr(e))
        raise Exception(str(sampleIndex))


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# aliLengthOnRef :
# Input : a CIGAR string
# Output : span of the alignment on the reference sequence, ie number of bases
# consumed by the alignment on the reference
def aliLengthOnRef(cigar):
    length = 0
    # only count CIGAR operations that consume the reference sequence, see CIGAR definition
    # in SAM spec available here: https://samtools.github.io/hts-specs/
    match = re.findall(r"(\d+)[MDN=X]", cigar)
    for op in match:
        length += int(op)
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
        exonNCLs[chrom] = ncl
    return(exonNCLs)


####################################################
# Qname2ExonCount :
# Given data representing all alignments for a single qname (must all map
# to the same chrom):
# - apply QC filters to ignore qnames that are impossible to interpret;
# - identify the genomic intervals that are covered by the sequenced fragment;
# - identify exons overlapped by these intervals, and increment their count.
# Args:
#   - 4 lists of ints for F and R strand positions (start and end), in "read" order
#   - the NCL for the chromosome where current alignments map
#   - the counts vector to update (1D numpy int array)
#   - the maximum accepted gap length between a pair of mate reads, pairs separated by
#     a longer gap are ignored (putative structural variant or alignment error)
# This function updates countsVec, and returns a 3-element list [BP1, BP2, CNVTYPE] if
# the QNAME supports the presence of a CNV (CNVTYPE == 'DEL' or 'DUP') with breakpoints
# BP1 & BP2 (ie a read spans the breakpoints), None otherwise.
def Qname2ExonCount(startFList, endFList, startRList, endRList, exonNCL, countsVec, maxGap):
    # skip Qnames that don't have alignments on both strands
    if (len(startFList) == 0) or (len(startRList) == 0):
        return
    # skip Qnames that have at least 3 alignments on a strand
    if (len(startFList) > 2) or (len(startRList) > 2):
        return

    # declare list that will be returned
    breakPoints = None

    # if we have 2 alis on one strand and they overlap: merge them
    # (there could be a short indel or other SV, not trying to detect
    # this but we still cover this genomic region with this qname)
    if (len(startFList) == 2) and (min(endFList) > max(startFList)):
        startFList = [min(startFList)]
        endFList = [max(endFList)]
    if (len(startRList) == 2) and (min(endRList) > max(startRList)):
        startRList = [min(startRList)]
        endRList = [max(endRList)]

    #######################################################
    # identify genomic regions covered by the fragment
    #######################################################
    # fragStarts, fragEnds: start and end coordinates of the genomic intervals
    # covered by this fragment.
    # There is usually one (1F1R overlapping) or 2 intervals (1F1R not
    # overlapping) , but if the fragment spans a DEL there can be up to 3
    fragStarts = []
    fragEnds = []

    ##########################
    # 1F1R
    if (len(startFList) == 1) and (len(startRList) == 1):
        if (startFList[0] < endRList[0]):
            # face to face
            if (endFList[0] >= startRList[0]):
                # overlap => merge into a single interval
                fragStarts = [min(startFList[0], startRList[0])]
                fragEnds = [max(endFList[0], endRList[0])]
            elif (startRList[0] - endFList[0] <= maxGap):
                # no overlap but small gap between reads, assume no-CNV with library
                # fragment slightly longer than readLength*2 -> 2 intervals
                fragStarts = [startFList[0], startRList[0]]
                fragEnds = [endFList[0], endRList[0]]
            else:
                # large gap between reads: could be a DEL but without spanning reads,
                # so no clear evidence -> ignore qname
                return
        else:
            # back to back, could be a DUP but no spanning reads -> ignore qname
            return

    ##########################
    # 2F1R
    elif (len(startFList) == 2) and (len(startRList) == 1):
        if (startFList[1] <= endRList[0]) and (endFList[1] >= startRList[0]):
            # second F ali overlaps with R ali, build an interval by merging them
            fragStarts = [min(startFList[1], startRList[0])]
            fragEnds = [max(endFList[1], endRList[0])]
        elif (startFList[1] <= endRList[0]) and (startRList[0] - endFList[1] <= maxGap):
            # second F ali and R ali are face-to-face, they don't overlap but gap
            # is small -> build 2 intervals
            fragStarts = [startFList[1], startRList[0]]
            fragEnds = [endFList[1], endRList[0]]
        else:
            # second F ali and R ali are face-to-face but far apart, or they are back-to-back,
            # in both cases we can't interpret -> ignore qname
            return

        if (startFList[0] < startFList[1]):
            # F alis are in read order: spanning a DEL?
            if (endFList[0] < fragStarts[0]):
                # indeed this looks like a DEL, with first F ali an interval on its own (on the left)
                fragStarts.insert(0, startFList[0])
                fragEnds.insert(0, endFList[0])
                # this is a suspected DEL with BPs: fragEnds[0], fragStarts[1]
                breakPoints = [fragEnds[0], fragStarts[1], 'DEL']
            else:
                # Both F alis are overlapped by R ali, doesn't make sense, ignore this qname
                return
        else:
            # F alis are inverted, ie beginning of read aligns after end of read on
            # the genome: spanning a DUP?
            if (fragEnds[-1] < startFList[0]):
                # indeed this looks like a DUP, with first F ali (==start of F read) an interval on
                # its own (on the right)
                fragStarts.append(startFList[0])
                fragEnds.append(endFList[0])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            elif (fragStarts[-1] < endFList[0]):
                # first F ali overlaps R ali, very small DUP and/or large fragment? suspicious, but merge them
                fragStarts[-1] = min(fragStarts[-1], startFList[0])
                fragEnds[-1] = max(fragEnds[-1], endFList[0])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            else:
                # F alis suggest a DUP but R ali doesn't agree, ignore this qname
                return

    ##########################
    # 1F2R
    elif (len(startFList) == 1) and (len(startRList) == 2):
        if (startFList[0] <= endRList[0]) and (endFList[0] >= startRList[0]):
            # first R ali overlaps with F ali, build an interval by merging them
            fragStarts = [min(startFList[0], startRList[0])]
            fragEnds = [max(endFList[0], endRList[0])]
        elif (startFList[0] <= endRList[0]) and (startRList[0] - endFList[0] <= maxGap):
            # first R ali and F ali are face-to-face, they don't overlap but gap
            # is small -> build 2 intervals
            fragStarts = [startFList[0], startRList[0]]
            fragEnds = [endFList[0], endRList[0]]
        else:
            # first R ali and F ali are face-to-face but far apart, or they are back-to-back,
            # in both cases we can't interpret -> ignore qname
            return

        if (startRList[0] < startRList[1]):
            # R alis are in read order: spanning a DEL?
            if (fragEnds[-1] < startRList[1]):
                # indeed this looks like a DEL, with second R ali an interval on its own (on the right)
                fragStarts.append(startRList[1])
                fragEnds.append(endRList[1])
                # this is a suspected DEL with BPs: fragEnds[-2], fragStarts[-1]
                breakPoints = [fragEnds[-2], fragStarts[-1], 'DEL']
            else:
                # Both R alis are overlapped by F ali, doesn't make sense, ignore this qname
                return
        else:
            # R alis are inverted: spanning a DUP?
            if (endRList[1] < fragStarts[0]):
                # indeed this looks like a DUP, with second R ali (==start of R read) an interval on
                # its own (on the left)
                fragStarts.insert(0, startRList[1])
                fragEnds.insert(0, endRList[1])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            elif (fragStarts[0] < endRList[1]):
                # first F ali overlaps R ali, very small DUP and/or large fragment? suspicious, but merge them
                fragStarts[0] = min(fragStarts[0], startRList[1])
                fragEnds[0] = max(fragEnds[0], endRList[1])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            else:
                # R alis suggest a DUP but F ali doesn't agree, ignore this qname
                return

    ##########################
    # 2F2R
    elif(len(startFList) == 2) and (len(startRList) == 2):
        if (startFList[0] < startFList[1]) and (startRList[0] < startRList[1]):
            # both F and R pairs of alis are in order: spanning a DEL?
            if ((startFList[0] <= endRList[0]) and (endFList[0] >= startRList[0]) and
                (startFList[1] <= endRList[1]) and (endFList[1] >= startRList[1])):
                # OK: first F+R reads overlap and same for second pair of F+R reads -> merge each pair
                fragStarts = [min(startFList[0], startRList[0]), min(startFList[1], startRList[1])]
                fragEnds = [max(endFList[0], endRList[0]), max(endFList[1], endRList[1])]
                # this is a suspected DEL with BPs: fragEnds[0], fragStarts[1]
                breakPoints = [fragEnds[0], fragStarts[1], 'DEL']
            else:
                # at least one pair of F+R reads doesn't overlap, how? ignore qname
                return

        elif (startFList[1] < startFList[0]) and (startRList[1] < startRList[0]):
            # both F and R pairs of alis are inverted: spanning a DUP?
            if ((startFList[1] <= endRList[1]) and (endFList[1] >= startRList[1]) and
                (startFList[0] <= endRList[0]) and (endFList[0] >= startRList[0])):
                # OK: first F+R reads overlap and same for second pair of F+R reads -> merge each pair
                fragStarts = [min(startFList[1], startRList[1]), min(startFList[0], startRList[0])]
                fragEnds = [max(endFList[1], endRList[1]), max(endFList[0], endRList[0])]
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            else:
                # at least one pair of F+R reads doesn't overlap, how? ignore qname
                return

        else:
            # one pair is in order but the other is not, doesn't make sense
            return

    #######################################################
    # find exons overlapped by the fragment and increment counters
    #######################################################
    # we want to increment countsVec at most once per exon, even if
    # we have several intervals that overlap the same exon
    exonsSeen = []
    for fi in range(len(fragStarts)):
        overlappedExons = exonNCL.find_overlap(fragStarts[fi], fragEnds[fi])
        for exon in overlappedExons:
            exonIndex = exon[2]
            if (exonIndex not in exonsSeen):
                exonsSeen.append(exonIndex)
                incrementCount(countsVec, exonIndex)
    return(breakPoints)


####################################################
# incrementCount:
# increment the counter in countsVec (1D numpy array) at index exonIndex
@numba.njit
def incrementCount(countsVec, exonIndex):
    countsVec[exonIndex] += 1
