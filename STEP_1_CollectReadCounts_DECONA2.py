#############################################################################################################
######################################## STEP1 Collect Read Count DECONA2 ###################################
#############################################################################################################
#STEPS:
#   A) Getopt user argument (ARGV) recovery
#   B) Checking that the mandatory parameters are presents
#   C) Checking that the parameters actually exist and processing
#       concerns the creation of a list containing the bams paths
#       based on the selection of either --bams or --bams-from arguments.
#   D) Parsing exonic intervals bed
#       "processBed" function used to check the integrity of the file, padding the intervals
#        +-10 bp (currently), sorting the positions on the 4 columns (1:CHR,2:START,3:END,4:Exon_ID)
#   E) Creating NCL for each chromosome
#       "exonDict" function used from the ncls module, creation of a dictionary :
#        key = chromosome , value : NCL object
#   F) Parsing old counts file (.tsv) if exist else new count Dataframe creation 
#       "parseCountsFile" function used to check the integrity of the file against the previously
#        generated bed file. It also allows to check if the pre-existing counts are in a correct format.
#   G) Definition of a loop for each BAM files and the reads counting.
#       "countFrags" function : 
#           -allows to sort alignments by BAM QNAME and to filter them on specific flags (see conditions in the script).
#            Realisation by samtool v1.9 (using htslib 1.9)
#           -also extracts the information for each unique QNAME. The information on the end position and the length of
#           the alignments are not present the "AliLengthOnRef" function retrieves them.
#           -the Qname informations is sent to the "Qname2ExonCount" function to perform the fragment count. 
#           -A check of the results is also performed.
#   This step completes the count dataframe.
#   Once all samples have been processed, this dataframe is saved.

#############################################################################################################
################################ Loading of the modules required for processing #############################
#############################################################################################################
# Python Modules
import sys
import getopt
import logging
import os
import numpy as np # numpy array objects
import re
# nested containment lists, similar to interval trees but faster (https://github.com/biocore-ntnu/ncls)
from ncls import NCLS
import subprocess # run samtools
import tempfile

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
# Output : returns a numpy structured array with columns CHR START END EXON_ID, and where:
# - a padding is added to exon coordinates (ie -padding for START and +padding for END, current padding=10bp)
# - exons are sorted by CHR, then START, then END, then EXON_ID
def processBed(bedFile):
    # number of bps used to pad the exon coordinates
    padding=10
    bedname=os.path.basename(bedFile)
    if not os.path.isfile(bedFile):
        logger.error("BED file %s doesn't exist",bedFile)
        sys.exit(1)
    try:
        exons=np.genfromtxt(bedFile,
                            dtype=None,
                            encoding=None,
                            names=('CHR','START','END','EXON_ID'))
        # compression == 'infer' by default => auto-works whether bedFile is gzipped, bgzipped or not
    except Exception as e:
        logger.error("error parsing BED file %s: %s", bedFile, e)
        sys.exit(1)

    ###############################
    ##### Sanity Check
    ###############################   
    if not np.issubdtype(exons["CHR"].dtype, np.str_): #data type numpy void 
        logger.error("In BED file %s, first column 'CHR' should be a string but numpy sees them as %s",
                     bedname,exons["CHR"].dtype)
        sys.exit(1)   
    
    if (exons["START"].dtype!=int) or (exons["END"].dtype!=int) :
        logger.error("In BED file %s,columns 2-3 'START'-'END' should be ints but numpy sees them as %s - %s",
                     bedname,exons["START"].dtype,exons["END"].dtype)
        sys.exit(1)
        
    if not np.issubdtype(exons["EXON_ID"].dtype, np.str_):
        logger.error("In BED file %s,4th column 'EXON_ID' should be a string but numpy sees them as %s",
                     bedname,exons["EXON_ID"].dtype)
        sys.exit(1)    
   
    #################
    ### Create temp array for padding and sorting 
    #################
    # 5 columns : CHR, START, END, EXON_ID and CHR_NUM, CHR_NUM is an
    # int version of CHR (for sorting)
    dt=np.dtype({'names':('CHR','START','END','EXON_ID','CHR_NUM'),
                 'formats':(exons["CHR"].dtype,int,int,exons["EXON_ID"].dtype,int)})
    ProcessArray=np.empty(len(exons), dtype=dt)
    
    # CHR -> CHR_NUM dictionary
    translateCHRDict={}

    # temp dict containing all non-numeric chrs, value is CHR and key is
    # the CHR stripped of 'chr' if present (e.g 'Y'->'chrY')
    remainingCHRs={}

    #maxChr: max int value in CHR column (after stripping 'chr' if present)
    maxCHR=0

    #dictionary checking that the EXON_ID are unique(key='EXON_ID', value=1) 
    exonIDDict={}

    for line in range(len(exons)):        
        #############################
        ##### Fill in the first 4 columns 
        #############################
        #CHR column: copy
        ProcessArray[line]["CHR"]=exons[line]["CHR"] 
        
        ###########
        #START and END columns: pad
        if (exons[line]["START"] < 0) or (exons[line]["END"] < 0):
            logger.error("In BED file %s, columns 2 and/or 3 contain negative values in line : ", bedname, line)
            sys.exit(1)
            
        ProcessArray[line]["START"] = exons[line]["START"] - padding
        # never negative
        if ProcessArray[line]["START"] < 0:
            ProcessArray[line]["START"] = 0
        ProcessArray[line]["END"] = exons[line]["END"] + padding
        
        ###########
        #EXON_ID column: copy and check that each ID is unique
        currentID = exons[line]["EXON_ID"]
        if (currentID in exonIDDict):
            logger.error("In BED file %s, EXON_ID (4th column) %s is not unique", bedname, currentID)
            sys.exit(1)
        else:
            ProcessArray[line]["EXON_ID"] = currentID
            exonIDDict[currentID] = 1

        #############################
        ##### CHR_NUM preparation
        #############################
        #we want ints so we remove chr prefix from CHR column if present
        currentCHR = exons[line]["CHR"]
        currentCHRNum = currentCHR.replace("chr","",1)
          
        if currentCHRNum.isdigit():
            currentCHRNum=int(currentCHRNum)
            translateCHRDict[currentCHR] = currentCHRNum
            if maxCHR<currentCHRNum:
                maxCHR=currentCHRNum
        else:
            #non-numeric chromosome: save in remainingChRs for later
            remainingCHRs[currentCHRNum]=currentCHR
            
    ###############
    #### Non-numerical chromosome conversion to int
    ###############
    #replace non-numerical chromosomes by maxCHR+1, maxCHR+2 etc
    increment=1
    # first deal with X, Y, M/MT in that order
    for chr in ["X","Y","M","MT"]:
        if chr in remainingCHRs:
            translateCHRDict[remainingCHRs[chr]] = maxCHR+increment
            increment+=1
            del remainingCHRs[chr]
    # now deal with any other non-numeric CHRs, in alphabetical order
    for key in sorted(remainingCHRs):
        translateCHRDict[remainingCHRs[key]] = maxCHR+increment
        increment+=1
        
    ############### 
    #### finally we can fill CHR_NUM column
    ###############
    for line in range(len(ProcessArray)):
        ProcessArray[line]["CHR_NUM"] = translateCHRDict[ProcessArray[line]["CHR"]]

    ############### 
    #### Sorting and recovery of the first four columns
    ###############    
    # sort by CHR_NUM, then START, then END, then EXON_ID
    exonsSort=np.sort(ProcessArray,order=['CHR_NUM','START','END','EXON_ID'])

    # delete the tmp column, and return result
    exons=exonsSort[["CHR","START","END","EXON_ID"]]
    return(exons)


####################################################
#Create nested containment lists (similar to interval trees but faster), one per
# chromosome, storing the exons
#Input : numpy array of exons, as returned by processBed
#Output: dictionary(hash): key=chr, value=NCL
def createExonDict(exons):
    exonDict={}
    listCHR=np.unique(exons["CHR"])
    for chr in listCHR:
        index=np.where(np.in1d(exons["CHR"],chr))[0]
        exonsOnChr=exons[index]
        ncls = NCLS(exonsOnChr["START"], exonsOnChr["END"], index)
        exonDict[chr]=ncls
    return(exonDict)

#################################################
# parseCountsFile:
# Input:
#   - countsFile is a tsv file (with path), including column titles, as
#     produced by this program
#   - exons is a numpy.array holding exon definitions, padded and sorted,
#     as returned by processBed
#   - countsArray is an empty int numpy array (dim=NbExons*NbsampleToProcess)
#
# -> make sure countsFile was produced with the same BED as exons, else die;
# -> for any sample present in both countsFile and countsArray, fill the countsArray column
#    by copying data from countsFile
def parseCountsFile(countsFile,exons,countsArray):
    try:
        counts=open(countsFile,"r")
    except Exception as e:
        logger.error("Opening provided countsFile %s: %s", countsFile, e)
        sys.exit(1)

    ######################
    # parse header from (old) countsFile
    oldHeader = counts.readline().rstrip().rsplit("\t")

    # fill old2new to identify countsArray samples that are already in countsFile:
    # old2new is a vector, same size as oldHeader, value old2new[old] is:
    # the index of sample oldHeader[old] in countsArray (if present);
    # -1 otherwise, ie if oldHeader[old] is one of the exon definition columns "CHR",
    #    "START", "END", "EXON_ID" or if it's a sample absent from countsArray
    old2new = np.array([-1]*len(oldHeader))

    for oldIndex in range(len(oldHeader)):
        for newIndex in range(len(countsArray.dtype.names)):
            if oldHeader[oldIndex] == countsArray.dtype.names[newIndex]:
                old2new[oldIndex] = newIndex
                break

    ######################
    # parse data lines from countsFile
    lineIndex = 0            
    for line in counts:
        splitLine=line.rstrip().rsplit("\t")

        ######################
        ####### Comparison with exons columns
        ######################
        if splitLine[0]!=exons[lineIndex]["CHR"]:
            logger.error("'CHR' column differs between BED file and previous countsFile at line %i", lineIndex)
            sys.exit(1)

        if (int(splitLine[1])!=exons[lineIndex]["START"]) or (int(splitLine[2])!=exons[lineIndex]["END"]): 
            logger.error("'START' or 'END' values differ between BED file and previous countsFile at line %i", lineIndex)
            sys.exit(1)

        if splitLine[3]!=exons[lineIndex]["EXON_ID"]:
            logger.error("'EXON_ID' column differs between BED file and previous countsFile at line %i", lineIndex)
            sys.exit(1)

        #####################
        ###### Fill countsArray with old fragment count data
        #####################
        for oldIndex in range(len(old2new)):
            if old2new[oldIndex]!=-1:
                countsArray[lineIndex][old2new[oldIndex]] = int(splitLine[oldIndex])

        lineIndex+=1

####################################################
# countFrags :
# Count the fragments in bamFile that overlap each exon described in exonDict.
# Arguments:
#   - sample identifier [str]
#   - a bam file (with path)
#   - the dictionary of exon NCLs
#   - a tmp dir with fast RW access and enough space for samtools sort
#   - the number of cpu threads that samtools can use
#   - the numpy array to fill in with count data
#
# Raises an exception if something goes wrong
def countFrags(sampleName,bamFile,exonDict,tmpDir,num_threads,countsArray):
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
            line=line.rstrip('\r\n')
            align=line.split('\t')

            # skip ali if not on main chr
            if not mainChr.match(align[2]): 
                continue

            ######################################################################
            # If we are done with previous qname: process it and reset accumulators
            if (qname!=align[0]) and (qname!=""):  # align[0] is the qname
                if not qBad:
                    Qname2ExonCount(sampleName,qchrom,qstartF,qendF,qstartR,qendR,exonDict,countsArray)
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
            Qname2ExonCount(sampleName,qchrom,qstartF,qendF,qstartR,qendR,exonDict,countsArray)

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
#   -sample identifier [str]
#   -chr [str]
#   -4 lists of ints for F and R strand positions (start and end)
#   -the dictionary containing the NCLs for each chromosome
#   -the numpy array to fill in with count data, appropriate counts will be incremented
# Nothing is returned, this function just updates countsArray
def Qname2ExonCount(sampleName,chromString,startFList,endFList,startRList,endRList,exonDict,countsArray):
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
    # we want to increment vecExonCount at most once per exon, even if
    # we have two intervals and they both overlap the same exon
    exonsSeen = []
    for idx in range(len(Frag) // 2):
        overlappedExons = RefNCL.find_overlap(Frag[2*idx],Frag[2*idx+1])
        for exon in overlappedExons:
            exonIndex = int(exon[2])
            if (exonIndex in exonsSeen):
                continue
            else:
                exonsSeen.append(exonIndex)
                countsArray[sampleName][exonIndex] += 1

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    logger.debug("Entering main()")
    ##########################################
    # A) Getopt user argument (ARGV) recovery
    bams=""
    bamsFrom=""
    bedFile=""
    # default setting ARGV 
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
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--bams"):
            bams=value     
        elif opt in ("--bams-from"):
            bamsFrom=value
        elif opt in ("--bed"):
            bedFile =value
        elif opt in ("--counts"):
            countsFile=value
        elif opt in ("--tmp"):
            tmpDir=value
        elif opt in ("--threads"):
            threads=int(value)
        else:
            sys.exit("ERROR : Programming error. Unhandled option "+opt+".\n")

    #####################################################
    # B) Checking that the mandatory parameters are presents
    if (bams=="" and bamsFrom=="") or (bams!="" and bamsFrom!=""):
        sys.exit("ERROR : You must use either --bams or --bams-from but not both.\n"+usage)
    if bedFile=="":
        sys.exit("ERROR : You must use --bedFile.\n"+usage)

    #####################################################
    # C) Checking that the parameters actually exist and processing
    # bamsTmp is user-supplied and may have dupes
    bamsTmp=[]
    # bamsNoDupe: tmp dictionary for removing dupes if any: key==bam, value==1
    bamsNoDupe={}
    # bamsToProcess, with any dupes removed
    bamsToProcess=[]
    if bams!="":
        bamsTmp=bams.split(",")
    elif bamsFrom!="":
        if not os.path.isfile(bamsFrom):
            sys.exit("ERROR : bams-from file "+bamsFrom+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            bamListFile=open(bamsFrom,"r")
            for line in bamListFile:
                line = line.rstrip('\n')
                bamsTmp.append(line)
    else:
        sys.exit("ERROR : bams and bamsFile both empty, IMPOSSIBLE")

    # Check that all bams exist and that there aren't any duplicates
    for b in bamsTmp:
        if not os.path.isfile(b):
            sys.exit("ERROR : BAM "+b+" doesn't exist. Try "+scriptName+" --help.\n")
        elif b in bamsNoDupe:
            logger.warning("BAM "+b+" specified twice, ignoring the dupe")
        else:
            bamsNoDupe[b]=1
            bamsToProcess.append(b)

    if (countsFile!="") and (not os.path.isfile(countsFile)):
        sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n") 

    if not os.path.isdir(tmpDir):
        sys.exit("ERROR : tmp directory "+tmpDir+" doesn't exist. Try "+scriptName+" --help.\n")

    if (threads<=0):
        sys.exit("ERROR : number of threads "+str(threads)+" must be positive. Try "+scriptName+" --help.\n")

    ######################################################
    # D) Parsing exonic intervals bed
    logger.debug("starting processBed()")
    exons=processBed(bedFile)

    ######################################################
    # E) Creating NCLs for each chromosome
    logger.debug("starting createExonDict()")
    exonDict=createExonDict(exons)

    ############################################
    # F) Creating a numpy array to contain the counts results (new and old counts)
    logger.debug("creating empty countsArray")
    sampleNameList=[]
    for bam in bamsToProcess:
        sampleName=os.path.basename(bam)
        sampleName=sampleName.replace(".bam","")
        sampleNameList.append(sampleName)

    dt=np.dtype({'names':sampleNameList,
                 'formats': [np.int_]*len(sampleNameList)})
    countsArray=np.zeros(len(exons), dtype=dt) 

    ############################################
    # G) Parsing old counts file (.tsv) if provided
    if (countsFile!=""):
        logger.debug("starting parseCountsFile()")
        parseCountsFile(countsFile,exons,countsArray)
    
    #####################################################
    # H) Process each BAM
    logger.debug("starting to process each new BAM")
    sampleNameList=[] 
    for bam in bamsToProcess:
        sampleName=os.path.basename(bam)
        sampleName=sampleName.replace(".bam","")
        logger.info('Sample being processed : %s', sampleName)
        if np.sum(countsArray[sampleName])>0:
            logger.info('Sample %s already present in counts file, skipping it', sampleName)
            continue
        else:
            try:
                logger.debug("starting countFrags(%s)", sampleName)
                countFrags(sampleName, bam, exonDict, tmpDir, threads,countsArray)
            except Exception as e:
                logger.warning("Failed to count fragments for sample %s, skipping it - exception: %s",
                               sampleName, e)
                continue

    #####################################################
    # J) Print exon defs + counts to stdout
    logger.debug("printing results to stdout")
    toPrint = "\t".join(exons.dtype.names + countsArray.dtype.names)
    print(toPrint)
    for line in range(len(exons)):
        toPrint = "\t".join(map(str,exons[line]))
        toPrint += "\t" + "\t".join(map(str,countsArray[line]))
        print(toPrint)
    logger.debug("ALL DONE")
      

if __name__ =='__main__':
    main()
