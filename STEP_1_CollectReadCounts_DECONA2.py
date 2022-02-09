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
#        +-10 bp, sorting the positions on the 4 columns (1:CHR,2:START,3:END,4:Exon_ID)
#   E) Creating NCL for each chromosome
#       "exonDict" function used from the ncls module, creation of a dictionary :
#        key = chromosome , value : NCL object
#   F) Parsing old counts file (.tsv) if exist else new count Dataframe creation 
#       "parseCountFile" function used to check the integrity of the file against the previously
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
# 1) Python Modules
import sys
import getopt
import logging
import os
import pandas as pd # dataframe objects
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
# Output : returns the data as a pandas dataframe with column headers CHR START END EXON_ID,
#          and where:
# - a 10bp padding is added to exon coordinates (ie -10 for START and +10 for END)
# - exons are sorted by CHR, then START, then END, then EXON_ID
def processBed(bedFile):
    bedname=os.path.basename(bedFile)
    if not os.path.isfile(bedFile):
        logger.error("BED file %s doesn't exist.\n", bedFile)
        sys.exit(1)
    try:
        exons=pd.read_table(bedFile, header=None, sep="\t")
        # compression == 'infer' by default => auto-works whether bedFile is gzipped or not
    except Exception as e:
        logger.error("error parsing BED file %s: %s", bedname, e)
        sys.exit(1)
   #####################
    #I): Sanity Check
    #####################
    if len(exons.columns) != 4:
        logger.error("BED file %s should be a 4-column TSV (CHR, START, END, EXON_ID) but it has %i columns\n",
                     bedname, len(exons.columns))
        sys.exit(1)
    exons.columns=["CHR","START","END","EXON_ID"]
    #######################
    #CHR column
    if (exons["CHR"].dtype!="O"):
        logger.error("In BED file %s, first column 'CHR' should be a string but pandas sees it as %s\n",
                     bedname, exons["CHR"].dtype)
        sys.exit(1)
    # create CHR_NUM column for sorting: we want ints so we remove chr prefix if present
    if exons["CHR"].str.startswith('chr').all:
        exons['CHR_NUM'] = exons['CHR'].replace(regex=r'^chr(\w+)$', value=r'\1')
    else:
        exons['CHR_NUM']=exons['CHR']
    #######################
    #START and END columns
    if (exons["START"].dtype!=int) or (exons["END"].dtype!=int):
        logger.error("In BED file %s, columns 2-3 'START'-'END' should be ints but pandas sees them as %s - %s\n",
                     bedname, exons["START"].dtype, exons["END"].dtype)
        sys.exit(1)
    if (exons.START < 0).any() or (exons.END < 0).any():
        logger.error("In BED file %s, columns 2 and/or 3 contain negative values", bedname)
        sys.exit(1)
    #######################
    #transcript_id_exon_number column
    if (exons["EXON_ID"].dtype!="O"):
        logger.error("In BED file %s, 4th column 'EXON_ID' should be a string but pandas sees it as %s\n",
                     bedname, exons["EXON_ID"].dtype)
        sys.exit(1)
    # EXON_IDs must be unique
    if len(exons["EXON_ID"].unique()) != len(exons["EXON_ID"]):
        logger.error("In BED file %s, each line must have a unique EXON_ID (4th column)\n", bedname)
        sys.exit(1)
    
    #####################
    #II): Padding and sorting
    #####################
    # pad coordinates with +-10bp
    exons['START'] -= 10
    exons['END'] += 10
    
    # replace X Y M/MT (if present) by max(CHR)+1,+2,+3
    # maxChr must be the max among the int values in CHR_NUM column...
    maxChr = int(pd.to_numeric(exons['CHR_NUM'],errors="coerce").max(skipna=True))
    exons["CHR_NUM"]=exons["CHR_NUM"].replace(
        {'X': maxChr+1,
         'Y': maxChr+2,
         'M': maxChr+3,
         'MT': maxChr+3})
    
    # convert type of CHR_NUM to int and catch any errors
    try:
        exons['CHR_NUM'] = exons['CHR_NUM'].astype(int)
    except Exception as e:
        logger.error("While parsing BED file, failed converting CHR_NUM to int: %s", e)
        sys.exit(1)
    # sort by CHR_NUM, then START, then END, then EXON_ID
    exons.sort_values(by=['CHR_NUM','START','END','EXON_ID'], inplace=True, ignore_index=True)
    # delete the temp column, and return result
    exons.drop(['CHR_NUM'], axis=1, inplace=True)    
    return(exons)

####################################################
#Create nested containment lists (similar to interval trees but faster), one per
# chromosome, storing the exons
#Input : dataframe of exons, as returned by processBed
#Output: dictionary(hash): key=chr, value=NCL
def createExonDict(exons):
    exonDict={}
    listCHR=list(exons.CHR.unique())
    for chr in listCHR:
        exonsOnChr=exons.loc[exons["CHR"]==chr]
        ncls = NCLS(exonsOnChr["START"], exonsOnChr["END"], exonsOnChr.index)
        exonDict[chr]=ncls
    return(exonDict)

#################################################
# parseCountFile:
#Input:
#   - countFile is a tsv file (with path), including column titles, as
#     produced by this program
#   - exons is a dataframe holding exon definitions, padded and sorted,
#     as returned by processBed
#
#-> Parse countFile into a dataframe (will be returned)
#-> Check that the first 4 columns are identical to exons,
#    otherwise die with an error.
#-> check that the samples counts columns are [int] type.
# 
#Output: returns the Frag count results as a pandas dataframe with column 
# headers CHR START END EXON_ID,sampleName*n

def parseCountFile(countFile, exons):
    try:
        counts=pd.read_table(countFile,sep="\t")
    except Exception as e:
        logger.error("Parsing provided countFile %s: %s", countFile, e)
        sys.exit(1)
    #Check if data are identical: this checks geometry, column dtypes, and values
    if not (counts.iloc[:,0:4].equals(exons)):
        logger.error("first 4 columns in countFile %s differ from the BED (after padding and sorting), "+
                     "if you updated your transcript set (BED) you cannot re-use the previous countFile, "+
                     "in this case you must re-analyze all samples (ie don't use --counts)", countFile)
        sys.exit(1)
    #check that the count columns are ints
    for columnIndex in range(4,len(counts.columns)):
        if not counts.dtypes[columnIndex]==int:
            # just die: this shouldn't happen if countFile was made by us
            logger.error("in countFile %s, column for sample %s is not all ints\n",
                         countFile, counts.columns[columnIndex]);
            sys.exit(1)
    return(counts)

####################################################
# countFrags :
# Count the fragments in bamFile that overlap each exon described in exonDict.
# Arguments:
#   - a bam file (with path)
#   - the dictionary of exon NCLs
#   - the total number of exons
#   - a tmp dir with fast RW access and enough space for samtools sort
#   - the number of cpu threads that samtools can use
#
# Returns a vector containing the fragment counts, in the same order as in the bed
# file used to create the NCLs.
# Raises an exception if something goes wrong
def countFrags(bamFile, exonDict, nbOfExons, processTmpDir, num_threads):
    # We need to process all alignments for a given qname simultaneously
    # => ALGORITHM:
    # parse alignements from BAM, sorted by qname;
    # if qname didn't change -> just apply some QC and if AOK store the
    #   data we need in accumulators,
    # if the qname changed -> process accumulated data for the previous
    #   qname (with Qname2ExonCount), then reset accumulators and store
    #   data for new qname.

    # Result that will be returned: list containing the number of fragments
    # overlapping each exon, indexes correspond to the labels in the NCL
    vecExonCount=[0]*nbOfExons

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

    with tempfile.TemporaryDirectory(dir=processTmpDir) as SampleTmpDir:
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
                    Qname2ExonCount(qchrom,qstartF,qendF,qstartR,qendR,exonDict,vecExonCount)
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
            Qname2ExonCount(qchrom,qstartF,qendF,qstartR,qendR,exonDict,vecExonCount)

        # wait for samtools to finish cleanly and check return codes
        if (p1.wait() != 0):
            logger.error("in countFrags, while processing %s, the 'samtools sort' subprocess returned %s",
                         bamFile, p1.returncode)
            raise Exception("samtools-sort failed")
        if (p2.wait() != 0):
            logger.error("in countFrags, while processing %s, the 'samtools view' subprocess returned %s",
                         bamFile, p2.returncode)
            raise Exception("samtools-view failed")
        
        # AOK, return counts
        return(vecExonCount)     
    

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
#   -list of fragment counts for each exon, appropriate counts will be incremented
# Nothing is returned, this function just updates vecExonCount
def Qname2ExonCount(chromString,startFList,endFList,startRList,endRList,exonDict,vecExonCount):
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
                vecExonCount[exonIndex] += 1
 

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    ##########################################
    # A) Getopt user argument (ARGV) recovery
    bams=""
    bamsFrom=""
    bedFile=""
    # default setting ARGV 
    countFile=""
    tmpDir="/tmp/"
    threads=10 

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- 10bp padding).
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after
padding, subsequent columns (one per BAM) hold the counts. If a pre-existing counts file produced
by this program with the same BED is provided (with --counts), its content is copied and counting
is only performed for the new BAM(s).
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
            countFile=value
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
    bamsToProcess=[]
    if bams!="":
        bamsToProcess=bams.split(",")
    
    elif bamsFrom!="":
        if not os.path.isfile(bamsFrom):
            sys.exit("ERROR : bams-from file "+bamsFrom+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            bamListFile=open(bamsFrom,"r")
            for line in bamListFile:
                line = line.rstrip('\n')
                bamsToProcess.append(line)
    else:
        sys.exit("ERROR : bams and bamsFile both empty, IMPOSSIBLE")
    #Check that all bams exist
    for b in bamsToProcess:
        if not os.path.isfile(b):
            sys.exit("ERROR : BAM "+b+" doesn't exist. Try "+scriptName+" --help.\n")

    if (countFile!="") and (not os.path.isfile(countFile)):
        sys.exit("ERROR : countFile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n") 

    if not os.path.isdir(tmpDir):
        sys.exit("ERROR : tmp directory "+tmpDir+" doesn't exist. Try "+scriptName+" --help.\n")

    if (threads<=0):
        sys.exit("ERROR : number of threads "+str(threads)+" must be positive. Try "+scriptName+" --help.\n")

    ######################################################
    # D) Parsing exonic intervals bed
    exons=processBed(bedFile)
    nbOfExons=len(exons)

    ######################################################
    # E) Creating NCLs for each chromosome
    exonDict=createExonDict(exons)

    ############################################
    # F) Parsing old counts file (.tsv) if provided, else make a new dataframe
    if (countFile!=""):
        counts=parseCountFile(countFile,exons)
    else:
        counts=exons

    #####################################################
    # G) Process each BAM
    for bam in bamsToProcess:
        sampleName=os.path.basename(bam)
        sampleName=sampleName.replace(".bam","")
        logger.info('Sample being processed : %s', sampleName)

        if sampleName in list(counts.columns[4:]):
            logger.info('Sample %s already present in counts file, skipping it\n', sampleName)
            continue
        else:
            try:
                FragVec = countFrags(bam, exonDict, nbOfExons, tmpDir, threads)
            except Exception as e:
                logger.warning("Failed to count fragments for sample %s, skipping it - exception: %s\n",
                               sampleName, e)
                continue
            counts[sampleName]=FragVec

    counts.to_csv(sys.stdout,sep="\t", index=False)
      

if __name__ =='__main__':
    main()
