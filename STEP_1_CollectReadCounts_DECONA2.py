###############################################################################################
######################################## STEP1 Collect Read Count DECONA2 #####################
###############################################################################################
# Given a BED of exons and one or more BAM files, count the number of sequenced fragments
# from each BAM that overlap each exon (+- padding).
# Print results to stdout. 
# See usage for details.
###############################################################################################

import sys
import getopt
import logging
import os
import numpy as np # numpy arrays
import re
# nested containment lists, similar to interval trees but faster (https://github.com/biocore-ntnu/ncls)
from ncls import NCLS
import subprocess # run samtools
import tempfile
import gzip
import time

#####################################################################################################
################################ Logging Definition #################################################
#####################################################################################################
# set up logger
logger=logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.DEBUG)
# create console handler for STDERR
stderr = logging.StreamHandler(sys.stderr)
stderr.setLevel(logging.DEBUG)
#create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                                '%Y-%m-%d %H:%M:%S')
#add formatter to stderr handler
stderr.setFormatter(formatter)
#add stderr handler to logger
logger.addHandler(stderr)

#####################################################################################################
################################ Functions ##########################################################
#####################################################################################################

####################################################
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
        logger.error("Opening provided BED file %s: %s", bedFile, e)
        sys.exit(1)

    for line in bedFH:
        fields = line.rstrip().split("\t")
        #############################
        # sanity checks + preprocess data
        # need exactly 4 fields
        if len(fields) != 4 :
            logger.error("In BED file %s, a line doesn't have 4 fields, illegal: %s",
                         bedname, line)
            sys.exit(1)
        # START and END must be ints
        if fields[1].isdigit() and fields[2].isdigit():
            # OK, convert to actual ints and pad (but don't pad to negatives)
            fields[1] = max(int(fields[1]) - padding, 0)
            fields[2] = int(fields[2]) + padding
        else:
            logger.error("In BED file %s, columns 2-3 START-END must be ints but are not in: %s",
                         bedname, line)
            sys.exit(1)
        # EXON_ID must be unique
        if fields[3] in exonIDDict:
            logger.error("In BED file %s, EXON_ID (4th column) %s is not unique", bedname, fields[3])
            sys.exit(1)
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


####################################################
#Create nested containment lists (similar to interval trees but faster), one per
# chromosome, storing the exons
#Input : list of exons(ie lists of 4 fields), as returned by processBed
#Output: dictionary(hash): key=chr, value=NCL
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

#################################################
# parseCountsFile:
# Input:
#   - countsFile is a tsv file (with path), including column titles, as
#     produced by this program
#   - exons holds the exon definitions, padded and sorted, as returned
#     by processBed
#   - SOIs is a list of strings: the sample names of interest
#   - countsArray is an empty int numpy array (dim=NbExons*NbSOIs)
#
# -> make sure countsFile was produced with the same BED as exons, else die;
# -> for any sample present in both countsFile and SOIs, fill the sample's
#    column in countsArray by copying data from countsFile, and set
#    countsFilled[sample] to True
def parseCountsFile(countsFile,exons,SOIs,countsArray,countsFilled):
    try:
        counts=open(countsFile,"r")
    except Exception as e:
        logger.error("Opening provided countsFile %s: %s", countsFile, e)
        sys.exit(1)

    ######################
    # parse header from (old) countsFile
    oldHeader = counts.readline().rstrip().split("\t")

    # fill old2new to identify samples of interest that are already in countsFile:
    # old2new is a vector, same size as oldHeader, value old2new[old] is:
    # the index of sample oldHeader[old] in SOIs (if present);
    # -1 otherwise, ie if oldHeader[old] is one of the exon definition columns "CHR",
    #    "START", "END", "EXON_ID" or if it's a sample absent from SOIs
    old2new = np.array([-1]*len(oldHeader))
    for oldIndex in range(len(oldHeader)):
        for newIndex in range(len(SOIs)):
            if oldHeader[oldIndex] == SOIs[newIndex]:
                old2new[oldIndex] = newIndex
                countsFilled[newIndex] = True
                break

    ######################
    # parse data lines from countsFile
    lineIndex = 0
    for line in counts:
        splitLine=line.rstrip().split("\t")

        ####### Compare exon definitions
        if ((splitLine[0]!=exons[lineIndex][0]) or
            (not splitLine[1].isdigit()) or (int(splitLine[1])!=exons[lineIndex][1]) or
            (not splitLine[2].isdigit()) or (int(splitLine[2])!=exons[lineIndex][2]) or
            (splitLine[3]!=exons[lineIndex][3])) :
            logger.error("exon definitions disagree between previous countsFile %s and the provided BED file "+
                         "(after padding and sorting) at line index %i. If the BED file changed, a previous "+
                         "countsFile cannot be re-used: all counts must be recalculated from scratch", lineIndex)
            sys.exit(1)

        ###### Fill countsArray with old count data
        for oldIndex in range(len(old2new)):
            if old2new[oldIndex]!=-1:
                countsArray[lineIndex][old2new[oldIndex]] = int(splitLine[oldIndex])
        ###### next line
        lineIndex+=1

####################################################
# countFrags :
# Count the fragments in bamFile that overlap each exon described in exonDict.
# Arguments:
#   - a bam file (with path)
#   - the dictionary of exon NCLs
#   - a tmp dir with fast RW access and enough space for samtools sort
#   - the number of cpu threads that samtools can use
#   - the numpy array to fill in with count data
#   - the column index (in countsArray) corresponding to bamFile
#
# Raises an exception if something goes wrong
def countFrags(bamFile,exonDict,tmpDir,num_threads,countsArray,sampleIndex):
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

    with tempfile.TemporaryDirectory(dir=tmpDir) as SampleTmpDir:
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
                    Qname2ExonCount(qchrom,qstartF,qendF,qstartR,qendR,exonDict,countsArray,sampleIndex)
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
            Qname2ExonCount(qchrom,qstartF,qendF,qstartR,qendR,exonDict,countsArray,sampleIndex)

        # wait for samtools to finish cleanly and check return codes
        if (p1.wait() != 0):
            logger.error("in countFrags, while processing %s, the 'samtools sort' subprocess returned %s",
                         bamFile, p1.returncode)
            raise Exception("samtools-sort failed")
        if (p2.wait() != 0):
            logger.error("in countFrags, while processing %s, the 'samtools view' subprocess returned %s",
                         bamFile, p2.returncode)
            raise Exception("samtools-view failed")
 
    

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
#   -chr [str]
#   -4 lists of ints for F and R strand positions (start and end)
#   -the dictionary containing the NCLs for each chromosome
#   -the numpy array to fill, counts in column sampleIndex will be incremented
#   -the column index in countsArray corresponding to the current sample
# Nothing is returned, this function just updates countsArray
def Qname2ExonCount(chromString,startFList,endFList,startRList,endRList,exonDict,countsArray,sampleIndex):
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

    # Examine gap length between the two reads (negative if overlapping)
    # CAVEAT: hard-coded cutoff here, could be a problem if the sequencing
    # library fragments are often longer than maxGapBetweenReads+2*readLength
    maxGapBetweenReads = 1000
    if (len(startFList+startRList)==2): # 1F1R
        if (startRList[0] - endFList[0] > maxGapBetweenReads):
            # large gap between forward and reverse reads, could be a DEL but
            # we don't have reads spanning it -> insufficient evidence, skip qname
            return
    elif (len(startFList+startRList)==3): # 2F1R or 1F2R
        if (min(startRList)-max(endFList) > maxGapBetweenReads):
            # eg 2F1R: the R ali is too far from the rightmost F ali, so
            # a fortiori it is too far from the leftmost F ali => ignore qname
            return
        elif (max(startRList)-min(endFList) > maxGapBetweenReads):
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
    #Retrieve the corresponding NCL
    RefNCL=exonDict[chromString]
    # we want to increment countsArray at most once per exon, even if
    # we have two intervals and they both overlap the same exon
    exonsSeen = []
    for idx in range(len(Frag) // 2):
        overlappedExons = RefNCL.find_overlap(Frag[2*idx],Frag[2*idx+1])
        for exon in overlappedExons:
            exonIndex = exon[2]
            if (exonIndex in exonsSeen):
                continue
            else:
                exonsSeen.append(exonIndex)
                countsArray[exonIndex][sampleIndex] += 1

######################################################################################################
######################################## Main ########################################################
######################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    logger.info("starting to work")
    startTime = time.time()
    ##########################################
    # parse user-provided arguments
    # mandatory args
    bams=""
    bamsFrom=""
    bedFile=""
    # optional args with default values
    countsFile=""
    tmpDir="/tmp/"
    threads=10 

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- padding).
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after
padding and sorting, subsequent columns (one per BAM) hold the counts.
If a pre-existing counts file produced by this program with the same BED is provided (with --counts),
its content is copied and counting is only performed for the new BAM(s).
ARGUMENTS:
   --bams [str]: comma-separated list of BAM files
   --bams-from [str]: text file listing BAM files, one per line
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column 
                headerless tab-separated file, columns contain CHR START END EXON_ID)
   --counts [str] optional: pre-existing counts file produced by this program, content will be copied
   --tmp [str]: pre-existing dir for temp files, faster is better (eg tmpfs), default: """+tmpDir+"""
   --threads [int]: number of threads to allocate for samtools sort, default: """+str(threads)+"\n"

    
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","bams=","bams-from=","bed=","counts=","tmp=","threads="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--bams"):
            bams=value
            # bams is checked later, along with bamsFrom content
        elif opt in ("--bams-from"):
            bamsFrom=value
            if not os.path.isfile(bamsFrom):
                sys.exit("ERROR : bams-from file "+bamsFrom+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--bed"):
            bedFile=value
            if not os.path.isfile(bedFile):
                sys.exit("ERROR : bedFile "+bedFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--counts"):
            countsFile=value
            if not os.path.isfile(countsFile):
                sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n") 
        elif opt in ("--tmp"):
            tmpDir=value
            if not os.path.isdir(tmpDir):
                sys.exit("ERROR : tmp directory "+tmpDir+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--threads"):
            threads=int(value)
            if (threads<=0):
                sys.exit("ERROR : threads "+str(threads)+" must be a positive int. Try "+scriptName+" --help.\n")
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    # Check that the mandatory parameters are present
    if (bams=="" and bamsFrom=="") or (bams!="" and bamsFrom!=""):
        sys.exit("ERROR : You must use either --bams or --bams-from but not both.\n"+usage)
    if bedFile=="":
        sys.exit("ERROR : You must use --bedFile.\n"+usage)

    #####################################################
    # Check and clean up the provided list of BAMs
    # bamsTmp is user-supplied and may have dupes
    bamsTmp=[]
    # bamsNoDupe: tmp dictionary for removing dupes if any: key==bam, value==1
    bamsNoDupe={}
    # bamsToProcess, with any dupes removed
    bamsToProcess=[]
    # sample names stripped of path and .bam extension, same order as in bamsToProcess 
    sampleNames=[]

    if bams != "":
        bamsTmp=bams.split(",")
    else:
        bamsList = open(bamsFrom,"r")
        for bam in bamsList:
            bam = bam.rstrip()
            bamsTmp.append(bam)

    # Check that all bams exist and remove any duplicates
    for bam in bamsTmp:
        if not os.path.isfile(bam):
            sys.exit("ERROR : BAM "+bam+" doesn't exist. Try "+scriptName+" --help.\n")
        elif bam in bamsNoDupe:
            logger.warning("BAM "+bam+" specified twice, ignoring the dupe")
        else:
            bamsNoDupe[bam]=1
            bamsToProcess.append(bam)
            sampleName=os.path.basename(bam)
            sampleName=re.sub(".bam$","",sampleName)
            sampleNames.append(sampleName)


    ######################################################
    # Preparation:
    # parse exons from BED and create an NCL for each chrom
    exons=processBed(bedFile)
    exonDict=createExonDict(exons)
    thisTime = time.time()
    logger.debug("Done pre-processing BED, in %.2f s", thisTime-startTime)
    startTime = thisTime

    # countsArray[exonIndex][sampleIndex] will store the corresponding count.
    # order=F should improve performance, since we fill the array one column at a time.
    # dtype=uint32 should also be faster and totally sufficient to store the counts,
    # but in my tests it is MUCH slower than dtype=int !
    countsArray = np.zeros((len(exons),len(sampleNames)),dtype=int, order='F')
    # countsFilled: same size and order as sampleNames, value will be set 
    # to True iff counts were filled from countsFile
    countsFilled = np.array([False]*len(sampleNames))

    # fill countsArray with pre-calculated counts if countsFile was provided
    if (countsFile!=""):
        parseCountsFile(countsFile,exons,sampleNames,countsArray,countsFilled)
        thisTime = time.time()
        logger.debug("Done parsing old countsFile, in %.2f s", thisTime-startTime)
        startTime = thisTime

    #####################################################
    # Process each BAM
    # if countFrags fails for any BAMs, we have to remember their indexes
    # and only expunge them at the end, after exiting the for bamIndex loop
    # -> save their indexes in failedBams
    failedBams = []
    for bamIndex in range(len(bamsToProcess)):
        bam = bamsToProcess[bamIndex]
        sampleName = sampleNames[bamIndex]
        logger.info('Processing sample %s', sampleName)
        if countsFilled[bamIndex]:
            logger.info('Sample %s already filled from countsFile, skipping it', sampleName)
            continue
        else:
            try:
                countFrags(bam, exonDict, tmpDir, threads, countsArray, bamIndex)
                thisTime = time.time()
                logger.debug("Done processing BAM for %s, in %.2f s", sampleName, thisTime-startTime)
                startTime = thisTime
            except Exception as e:
                logger.warning("Failed to count fragments for sample %s, skipping it - exception: %s",
                               sampleName, e)
                failedBams.append(bamIndex)
                continue
    # now expunge samples for which countFrags failed
    for failedI in reversed(failedBams):
        del(sampleNames[failedI])
        countsArray = np.delete(countsArray,failedI,1)

    #####################################################
    # Print exon defs + counts to stdout
    toPrint = "CHR\tSTART\tEND\tEXON_ID\t"+"\t".join(sampleNames)
    print(toPrint)
    for exonIndex in range(len(exons)):
        toPrint = "\t".join(map(str,exons[exonIndex]))
        toPrint += "\t" + "\t".join(map(str,countsArray[exonIndex]))
        print(toPrint)
    thisTime = time.time()
    logger.debug("Done printing results, in %.2f s", thisTime-startTime)
    logger.info("ALL DONE")


if __name__ =='__main__':
    main()
