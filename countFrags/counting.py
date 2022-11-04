import sys
import os
import numpy as np # numpy arrays
import numba # make python faster
import re
# nested containment lists, similar to interval trees but faster (https://github.com/biocore-ntnu/ncls)
from ncls import NCLS
import subprocess # run samtools
import tempfile
import time
import logging

# set up logger, using inherited config
logger = logging.getLogger(__name__)

#############################################################
################ Function
#############################################################
# Create nested containment lists (similar to interval trees but faster), one per
# chromosome, storing the exons
# Input : list of exons(ie lists of 4 fields), as returned by processBed
# Output: dictionary(hash): key=chr, value=NCL
def createExonDict(exons):
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
    exonDict={}
    for chrom in starts.keys():
        ncls = NCLS(starts[chrom], ends[chrom], indexes[chrom])
        exonDict[chrom]=ncls
    return(exonDict)

####################################################
# countFrags :
# Count the fragments in bamFile that overlap each exon described in exonDict.
# Arguments:
#   - a bam file (with path)
#   - a tmp dir with fast RW access and enough space for samtools sort
#   - the maximum accepted gap length between reads pairs
#   - a list of lists contains exons information (dim=NbExons x [CHR, START, END, EXONID])
#  (columns types: [str,int,int,str])
#   - the number of cpu threads that samtools can use
#
# output :
# - 1-dimensional numpy array contains fragment counts[int] for one sample
# Raises an exception if something goes wrong
def countFrags(bamFile,tmpDir,maxGap,exons,num_threads):
    startTime = time.time()
    # Create NCLS (nested containment lists)
    # implement here because the output object is coded in Cython 
    # and the parallelization module (multiprocess) does not accept this 
    # object type as an argument.
    # Cleaner than defining a global variable.
    # Redefinition at each new sample but it's a fast step.
    exonDict=createExonDict(exons)

    thisTime = time.time()
    logger.debug("Done createExonDict for %s, in %.2f s",os.path.basename(bamFile), thisTime-startTime)
    startTime =thisTime

    # We need to process all alignments for a given qname simultaneously
    # => ALGORITHM:
    # parse alignements from BAM, sorted by qname;
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
    # qFirstOnForward==1 if the first-in-pair read of this qname is on the
    # forward reference strand, -1 if it's on the reverse strand, 0 if we don't yet know
    qFirstOnForward=0
    # qBad==True if qname contains alignments on differents chromosomes,
    # or if some of it's alis disagree regarding the strand on which the
    # first/last read-in-pair aligns
    qBad=False

    tmpDirObj=tempfile.TemporaryDirectory(dir=tmpDir)
    SampleTmpDir=tmpDirObj.name

    # To Fill:
    # 1D numpy array containing the sample fragment counts for all exons
    countsSample=np.zeros(len(exons), dtype=np.uint32)

    ############################################
    # Preprocessing:
    # Our algorithm needs to parse the alignements sorted by qname,
    # "samtools sort -n" allows this
    cmd1 ="samtools sort -n "+bamFile+" -@ "+str(num_threads)+" -T "+SampleTmpDir+" -O sam"
    p1 = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)
    
    # We can immediately filter out poorly mapped (low MAPQ) or low quality
    # reads (based on the SAM flags) with samtools view:
    # -q sets the minimum mapping quality;
    # -F excludes alignements where any of the specified bits is set, we want to filter out:
    #   4 0x4 Read Unmapped
    #   256 0x80 not primary alignment
    #   512 0x200 Read fails platform/vendor quality checks
    #   1024 0x400 Read is PCR or optical duplicate
    # Total Flag decimal = 1796
    # For more details on FLAGs read the SAM spec: http://samtools.github.io/hts-specs/
    cmd2 ="samtools view -q 20 -F 1796"
    p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout, stdout=subprocess.PIPE, universal_newlines=True)

    ############################################
    # Regular expression to identify alignments on the main chromosomes (no ALTs etc)
    mainChr=re.compile("^chr[\dXYM][\dT]?$|^[\dXYM][\dT]?$")

    ############################################
    # Main loop: parse each alignment
    for line in p2.stdout:
        align=line.rstrip().split('\t')

        # skip ali if not on main chr
        if not mainChr.match(align[2]): 
            continue

        ######################################################################
        # If we are done with previous qname: process it and reset accumulators
        if (qname!=align[0]) and (qname!=""):  # align[0] is the qname
            if not qBad:
                Qname2ExonCount(qstartF,qendF,qstartR,qendR,exonDict[qchrom],countsSample,maxGap)
            qname, qchrom = "", ""
            qstartF, qstartR, qendF, qendR = [], [], [], []
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
        if qchrom=="":
            qchrom=align[2]
        elif qchrom!=align[2]:
            qBad=True
            continue
        # else same chrom, don't modify qchrom

        #Retrieving flags for STRAND and first/second read
        currentFlag=int(align[1])
        #flag 16 the alignment is on the reverse strand
        currentStrand="F"
        if currentFlag&16 :
            currentStrand="R"

        # currentFirstOnForward==1 if according to this ali, the first-in-pair
        # read is on the forward strand, -1 otherwise
        currentFirstOnForward=-1
        #flag 64 the alignment is first in pair, otherwise 128
        if (((currentFlag&64) and (currentStrand=="F")) or 
            ((currentFlag&128) and (currentStrand=="R"))) :
            currentFirstOnForward=1
        if qFirstOnForward==0:
            # first ali for this qname
            qFirstOnForward=currentFirstOnForward
        elif qFirstOnForward!=currentFirstOnForward:
            qBad=True
            continue
        # else this ali agrees with previous alis for qname -> NOOP

        # START and END coordinates of current alignment
        currentStart=int(align[3])
        # END depends on CIGAR
        currentCigar=align[5]
        currentAliLength=AliLengthOnRef(currentCigar)
        currentEnd=currentStart+currentAliLength-1
        if currentStrand=="F":
            qstartF.append(currentStart)
            qendF.append(currentEnd)
        else:
            qstartR.append(currentStart)
            qendR.append(currentEnd)

    #################################################################################################
    # Done reading lines from BAM

    # process last Qname
    if not qBad:
        Qname2ExonCount(qstartF,qendF,qstartR,qendR,exonDict[qchrom],countsSample,maxGap)

    # wait for samtools to finish cleanly and check return codes
    if (p1.wait() != 0):
        logger.error("in countFrags, while processing %s, the 'samtools sort' subprocess returned %s",
                        bamFile, p1.returncode)
        raise Exception("samtools-sort failed")
    if (p2.wait() != 0):
        logger.error("in countFrags, while processing %s, the 'samtools view' subprocess returned %s",
                        bamFile, p2.returncode)
        raise Exception("samtools-view failed")

    # SampleTmpDir should get cleaned up automatically but sometimes samtools tempfiles
    # are in the process of being deleted => sync to avoid race
    os.sync()
    thisTime = time.time()
    logger.debug("Done countsFrag for %s, in %.2f s",os.path.basename(bamFile), thisTime-startTime)
    return(countsSample)

####################################################
# AliLengthOnRef :
#Input : a CIGAR string
#Output : span of the alignment on the reference sequence, ie number of bases
# consumed by the alignment on the reference
def AliLengthOnRef(CIGARAlign):
    length=0
    # only count CIGAR operations that consume the reference sequence, see CIGAR definition 
    # in SAM spec available here: https://samtools.github.io/hts-specs/
    match = re.findall(r"(\d+)[MDN=X]",CIGARAlign)
    for op in match:
        length+=int(op)
    return(length)

####################################################
# Qname2ExonCount :
# Given data representing all alignments for a single qname:
# - apply QC filters to ignore aberrant or unusual alignments / qnames;
# - identify the genomic positions that are putatively covered by the sequenced fragment,
#   either actually covered by a sequencing read or closely flanked by a pair of mate reads;
# - identify exons overlapped by the fragment, and increment their count.
# Inputs:
#   -4 lists of ints for F and R strand positions (start and end)
#   -the NCL for the chromosome where current alignments map
#   -the numpy array to fill, counts in exonIndex will be incremented
#   -the column index in countsArray corresponding to the current sample
#   - the maximum accepted gap length between reads pairs, pairs separated by a longer gap
#       are assumed to possibly result from a structural variant and are ignored
# Nothing is returned, this function just updates countsArray
def Qname2ExonCount(startFList,endFList,startRList,endRList,exonNCL,countsSample,maxGap):
    #######################################################
    # apply QC filters
    #######################################################
    # skip Qnames that don't have alignments on both strands
    if (len(startFList)==0) or (len(startRList)==0):
        return
    # skip Qnames that have at least 3 alignments on a strand
    if (len(startFList)>2) or (len(startRList)>2):
        return

    # if we have 2 alis on one strand and they overlap: merge them
    # (there could be a short DUP, not trying to detect this but we still
    # cover this genomic region with this qname)
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

    if (len(startFList+startRList)==2): # 1F1R
        if startFList[0] > endRList[0]:
            # alignments are back-to-back (SV? Dup? alignment error?)
            return
    # elif (len(startFList+startRList)==3)
    # 1F2R and 2F1R have either become 1F1R, or they can't have any back-to-back pair
    elif (len(startFList+startRList)==4): #2F2R
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
    if (len(startFList+startRList)==2): # 1F1R
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
    # Frag: genomic intervals covered by this fragment, 
    # as a list of (pairs of) ints: start1,end1[,start2,end2...]
    # There is usually one (1F1R overlapping) or 2 intervals (1F1R not 
    # overlapping) , but if the fragment spans a DEL there can be up to 3
    Frag=[]
    # algo: merge F-R pairs of alis only if they overlap
    if (len(startFList)==1) and (len(startRList)==1):
        if (startRList[0] - endFList[0] < 0): # overlap => merge
            Frag=[min(startFList + startRList), max(endFList + endRList)]
        else:
            Frag=[startFList[0], endFList[0], startRList[0], endRList[0]]

    elif (len(startFList)==2) and (len(startRList)==1): #2F1R
        if (startRList[0] < min(endFList)): 
            # leftmost F ali overlaps R ali => merge into a single interval (because we know
            # that the rightmost F ali overlaps the R ali
            Frag = [min(startRList + startFList), max(endRList + endFList)]
        else:
            # leftmost F ali is an interval by itself
            Frag = [min(startFList),min(endFList)]
            if (startRList[0] < max(endFList)):
                # rightmost F ali overlaps R ali => merge
                Frag = Frag + [min(max(startFList),startRList[0]), max(endFList + endRList)]
            else:
                # no overlap, add 2 more intervals
                Frag = Frag + [max(startFList), max(endFList), startRList[0], endRList[0]]

    elif (len(startFList)==1) and (len(startRList)==2): #1F2R
        if (max(startRList) < endFList[0]): 
            # rightmost R ali overlaps F ali => merge into a single interval
            Frag = [min(startRList + startFList), max(endRList + endFList)]
        else:
            # rightmost R ali is an interval by itself
            Frag = [max(startRList),max(endRList)]
            if (min(startRList) < endFList[0]):
                # leftmost R ali overlaps F ali => merge
                Frag = Frag + [min(startFList + startRList), max(min(endRList),endFList[0])]
            else:
                # no overlap, add 2 more intervals
                Frag = Frag + [min(startRList), min(endRList), startFList[0], endFList[0]]

    elif(len(startFList)==2) and (len(startRList)==2): #2F2R
        # we already checked that each pair of F-R alis overlaps
        Frag=[min(startFList + startRList), max(min(endFList),min(endRList)),
              min(max(startFList),max(startRList)), max(endFList + endRList)]

    #######################################################
    # find exons overlapped by the fragment and increment counters
    #######################################################
    # we want to increment countsArray at most once per exon, even if
    # we have two intervals and they both overlap the same exon
    exonsSeen = []
    for idx in range(len(Frag) // 2):
        overlappedExons = exonNCL.find_overlap(Frag[2*idx],Frag[2*idx+1])
        for exon in overlappedExons:
            exonIndex = exon[2]
            if (exonIndex in exonsSeen):
                continue
            else:
                exonsSeen.append(exonIndex)
                incrementCount(countsSample, exonIndex)
                # countsArray[exonIndex][sampleIndex] += 1
    
####################################################
# incrementCount:
# increment the counters in countsSample for the specified exon index
@numba.njit
def incrementCount(countsSample, exonIndex):
    countsSample[exonIndex]+=1
