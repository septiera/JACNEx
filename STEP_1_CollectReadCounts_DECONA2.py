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
#   E) Creating interval trees for each chromosome
#       "intervalTreeDictCreation" function used from the ncls module, creation of a dictionary :
#        key = chromosome , value : interval_tree
#   F) Parsing old counts file (.tsv) if exist else new count Dataframe creation 
#       "parseCountFile" function used to check the integrity of the file against the previously
#        generated bed file. It also allows to check if the pre-existing counts are in a correct format.
#   G) Definition of a loop for each BAM files and the reads counting.
#       "SampleCountingFrag" function : 
#           -allows to sort alignments by BAM QNAME and to filter them on specific flags (see conditions in the script).
#            Realisation by samtool v1.9 (using htslib 1.9)
#           -also extracts the information for each unique QNAME. The information on the end position and the length of
#           the alignments are not present the "ExtractAliLength" function retrieves them.
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
import pandas as pd #read,make,treat Dataframe object
import re  # regular expressions
from ncls import NCLS # generating interval trees (cf : https://github.com/biocore-ntnu/ncls)
import subprocess #spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import tempfile #manages the creation and deletion of temporary folders/files

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
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s',
                                '%Y-%m-%d %H:%M:%S')
#add formatter to stderr handler
stderr.setFormatter(formatter)
#add stderr handler to logger
logger.addHandler(stderr)

#####################################################################################################
################################ Functions ##########################################################
#####################################################################################################
def usage():
    sys.stderr.write("\nCOMMAND SUMMARY:\n"+
"Given a BED of exons and one or more BAM files, count the number of sequenced fragments\n"+
"from each BAM that overlap each exon.\n"+
"Script will print to stdout a new countFile in TSV format, copying the data from the pre-existing\n"+ 
"countFile if provided and adding columns with counts for the new BAM(s).\n\n"+
"OPTIONS:\n"+
"   --bams [str]: a BAM file or a BAMs list (with path)\n"+
"   --bams-from [str] : a text file where each line contains the path to a BAM.\n"
"   --bed [str]: a bed file (with path), possibly gzipped, containing exon definitions \n"+
"                formatted as CHR START END EXON_ID\n"+
"   --counts [str] optional: a pre-parsed count file (with path), old fragment count file  \n"+
"                                  to be completed if new patient(s) is(are) added. \n"
"   --tmp [str] optional: a temporary folder (with path),allows to save temporary files \n"+
"                         from samtools sort. By default, this is placed in '/tmp'.\n"+
"                         Tip: place the storage preferably in RAM.\n"+
"   --threads [int] optional: number of threads to allocate for samtools.\n"+
"                             By default, the value is set to 10.\n\n")

####################################################
#Exon intervals file parsing and preparing.
# Input : bedFile == a bed file (with path), possibly gzipped, containing exon definitions
#         formatted as CHR START END EXON_ID
#
# Output : returns the data as a pandas dataframe with column headers CHR START END EXON_ID,
#          and where:
# - a 10bp padding is added to exon coordinates (ie -10 for START and +10 for END)
# - exons are sorted by CHR, then START, then END, then EXON_ID
def processBed(PathBedToCheck):
    bedname=os.path.basename(PathBedToCheck)#simplifies messages in stderr
    if not os.path.isfile(PathBedToCheck):
        logger.error("Exon intervals file %s doesn't exist.\n",bedname)
        sys.exit()
    BedToCheck=pd.read_table(PathBedToCheck,header=None,sep="\t")
    #####################
    #I): Sanity Check
    #####################
    if len(BedToCheck.columns) == 4:
        BedToCheck.columns=["CHR","START","END","EXON_ID"]
        #######################
        #CHR column
        if (BedToCheck["CHR"].dtype=="O"): # Check the python object is a str
            if not BedToCheck["CHR"].str.startswith('chr').all: 
                BedToCheck['CHR_NUM'] =BedToCheck['CHR']
                BedToCheck["CHR"]="chr"+BedToCheck["CHR"].astype(str)
            else:
                BedToCheck['CHR_NUM'] =BedToCheck['CHR']
                BedToCheck['CHR_NUM'].replace(regex=r'^chr(\w+)$', value=r'\1', inplace=True)
        else:
            logger.error("The 'CHR' column doesn't have an adequate format. Please check it.\n"+
            "The column must be a python object [str].\n")
            sys.exit()
        #######################
        #Start and End column
        if (BedToCheck["START"].dtype=="int64") and (BedToCheck["END"].dtype=="int64"):
            if (len(BedToCheck[BedToCheck.START<=0])>0) or (len(BedToCheck[BedToCheck.END<=0])>0):
                logger.error("Presence of outliers in the START and END columns. Values <=0.")
                sys.exit()
        else:
            logger.error("'START' and/or 'END' columns are not ints")
            sys.exit()
        #######################
        #transcript_id_exon_number column
        if not (BedToCheck["EXON_ID"].dtype=="O"):
            logger.error("The 'EXON_ID' column doesn't have an adequate format. Please check it.\n"+
            "The column must be a python object.\n")
            sys.exit()
    else:
        logger.error("BedToCheck doesn't contains 4 columns:\n"+
        "CHR, START, END, EXON_ID.\n"+
        "Please check the file before launching the script.\n")
        sys.exit()
    #####################
    #II): Preparation
    #####################
    # pad coordinates with +-10bp
    BedToCheck['START'] -= 10
    BedToCheck['END'] += 10

    # replace X Y M by len(unique(CHR))+1,+2,+3 (if present)
    BedToCheck["CHR_NUM"]=BedToCheck["CHR_NUM"].replace(
        {'X':len(BedToCheck['CHR_NUM'].unique())+1,
        "Y": len(BedToCheck['CHR_NUM'].unique())+2,
        "M":len(BedToCheck['CHR_NUM'].unique())+3,
        "MT":len(BedToCheck['CHR_NUM'].unique())+3})

    # convert type of CHR_NUM to int and catch any errors
    try:
        BedToCheck['CHR_NUM'] = BedToCheck['CHR_NUM'].astype(int)
    except Exception as e:
        logger.error("Converting CHR_NUM to int: %s", e)
        sys.exit(1)
    # sort by CHR_NUM, then START, then END, then EXON_ID
    exons= BedToCheck.sort_values(by=['CHR_NUM','START','END','EXON_ID'])
    # delete the temp column, and return result
    exons.drop(['CHR_NUM'], axis=1, inplace=True)    
    return(exons)

####################################################
#This function uses the ncls module to create interval trees
#Input : bed file with exon interval
#Output: dictionary(hash): key=chr, values=interval_tree
def intervalTreeDictCreation(BedIntervalFile):
    DictIntervalTree={}
    listCHR=list(BedIntervalFile.CHR.unique())
    for processChr in listCHR:
        intervalProcessChr=BedIntervalFile.loc[BedIntervalFile["CHR"]==processChr]
        ncls = NCLS(intervalProcessChr["START"].astype(int),intervalProcessChr["END"].astype(int),intervalProcessChr.index)
        DictIntervalTree[processChr]=ncls
    return(DictIntervalTree)

#################################################
# parseCountFile:
#Input:
#   - countFile is a tsv file (with path), including column titles, as
#     specified previously
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
    if (len(counts)==len(exons)): # lines number comparison  
        #Type Check 
        if not (counts.dtypes["CHR"]=="O" and counts.dtypes["EXON_ID"]=="O"):
            logger.error("One or both of the 'CHR' and 'EXON_ID' columns are not in the correct format. Please check it.\n"
                        +"The column must be a python object [str]")
            sys.exit(1)
        elif not (counts.dtypes["START"]=="int64" and counts.dtypes["END"]=="int64"):
            logger.error("One or both of the 'START' and 'END' columns are not in the correct format. Please check it.\n"
                        +"The columns must contain integers.")
            sys.exit(1)
        #Check if data are identical
        elif not (counts['CHR'].isin(exons['CHR']).value_counts())[True]==len(exons):
            logger.error("'CHR' column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        elif not (counts['START'].isin(exons['START']).value_counts())[True]==len(exons):
            logger.error("'START' column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        elif not (counts['END'].isin(exons['END']).value_counts())[True]==len(exons):
            logger.error("END column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        elif not (counts['EXON_ID'].isin(exons['EXON_ID']).value_counts())[True]==len(exons):
            logger.error("'EXON_ID' column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        else: 
            #check that the old samples columns data is in [int] format.
            namesSampleToCheck=[]
            for columnIndex in range(5,len(counts.columns)):
                if not counts.iloc[:,columnIndex].dtypes=="int64":
                    namesSampleToCheck.append(counts.iloc[:,columnIndex].columns)
            if len(namesSampleToCheck)>0:
                logger.error("Columns in %s, sample(s) %s are not in [int] format.\n"+
                "Please check and correct these before trying again.", countFile,(",".join(namesSampleToCheck)))
                sys.exit(1)
    else:
        logger.error("Old counts file %s doesn't have the same lines number as the exonic interval file.\n"+
        "Transcriptome version has probably changed.\n"+
        "In this case the whole samples set re-analysis must be done.\n"+
        "Do not set the --counts option.",countFile)
        sys.exit(1)
    return(counts)

####################################################
# SampleCountingFrag :
#This function allows samples to be processed for fragment counting.
#It uses several other functions such as: ExtractAliLength and Qname2ExonCount.
#Input:
#   -the full path to the bam file
#   -the dictionary of exonic interval trees 
#   -a dataframe of the exonic intervals (for output)
#   -the path to the temporary file
#   -the number of cpu allocated for processing (used for samtools commands)

#Output: a vector containing the fragment counts sorted according to the  bed file indexes.

def SampleCountingFrag(bamFile,dictIntervalTree,intervalBed,processTmpDir, num_threads):
    with tempfile.TemporaryDirectory(dir=processTmpDir) as SampleTmpDir:
        numberExons=len(intervalBed)
        ############################################
        # I] Pretreatments on the sample bam.
        # Samtools commands line :
        # The bam should be sorted by Qname to facilitate the fragments counting (-n option).
        # -@ allows the process to be split across multiple cores.
        # -T is used to store the process temporary files in the ram memory.
        # -O indicates the stdout result must be in sam format.
        cmd1 ="samtools sort -n "+bamFile+" -@ "+str(num_threads)+" -T "+SampleTmpDir+" -O sam"

        # Using samtools view, the following flag filters can be performed with the -F option:
        #   -4 0x4 Read Unmapped
        #   -256 0x80 not primary alignment
        #   -512 0x200 Read fails platform/vendor quality checks
        #   -1024 0x400 Read is PCR or optical duplicate
        # Total Flag decimal = 1796
        # The command also integrates a filtering on the mapping quality here fixed at 20 (-q option).
        # By default the stdout is in sam format
        cmd2 ="samtools view -F 1796 -q 20 -@ "+str(num_threads)
        # Orders execution
        p1 = subprocess.Popen(cmd1.split(),stdout=subprocess.PIPE,bufsize=1,encoding='ascii')
        p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout,stdout=subprocess.PIPE,bufsize=1,encoding='ascii')

        # III] Outpout variables initialization
        #list containing the fragment counts for the exons.
        vecExonCount=[0]*numberExons #The indexes correspond to the exon order in the "intervalBed".

        #Process control : skip mention corresponds to the Qnames removal for fragments counts
        #Aligments = alignements count, QN= qnames counts , Frag= fragments counts
        keys=["AlignmentsInBam",
            "AlignmentsOnMainChr",
            "QNProcessed",  #sum of all folowing keys => control
            "QNAliOnDiffChrSkip",
            "QNSameReadDiffStrandSkip",
            "QNSingleReadSkip",
            "QNNbAliR&FBetween1-3", #sum of the following 7 keys => control
            "QN1F&1R",
            "QN2F&1R_2FOverlapSkip",
            "QN2F&1R_SA",
            "QN1F&2R_2ROverlapSkip",
            "QN1F&2R_SA",
            "QNAliGapLength>1000bpSkip",
            "QNAliBackToBackSkip",
            "QNAli2F&2R_SA", #sum of the following 2 keys => control
            "QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip",
            "QNAliUnsuitableCombination_aliRorF>3Skip",
            "FragOverlapOnTargetInterval", # equal to the sum of VecExonCount results
            "QNNoOverlapOnTargetIntervalSkip",
            "QNOverlapOnTargetInterval"]
        dictStatCount={ i : 0 for i in keys }

        #Variables to be processed for each Qname's alignment.
        qchrom="" # unique
        qname="" # unique
        qstartR=[] #complete list if same Qname and same strand.
        qstartF=[]
        qendR=[]
        qendF=[]
        qReads=0 # two bits : 01 First read was seen , 10 Second read was seen
        qFirstOnForward=-1
        qBad=False #current Qname contains ali on differents chromosomes (by default set to false)

        ############################################
        # IV] Regular expression definition
        mainChr=re.compile("^chr[\dXYM]\d?$")
        ############################################
        # V] Function Main loop
        #Browse the file sorted on the qnames and having undergone the appropriate filters
        for line in p2.stdout:
            line=line.rstrip('\r\n')
            align=line.split('\t')
            dictStatCount["AlignmentsInBam"]+=1

            #################################################################################################
            #A] Selection of information for the current alignment only if it is on main chr
            if not mainChr.match(align[2]): continue #TODO make this work for different naming conventions
            dictStatCount["AlignmentsOnMainChr"]+=1 #Only alignments on the main chromosomes are treated

            #################################################################################################
            #B] If we are done with previous qname: process it and reset accumulators
            if (qname!=align[0]) and (qname!=""):  # align[0] is the qname
                dictStatCount["QNProcessed"]+=1
                if not qBad:
                    Qname2ExonCount(qname,qchrom,qstartF,qendF,qstartR,qendR,qReads,dictIntervalTree,vecExonCount,
                                    dictStatCount)
                qchrom=""
                qname=""
                qstartR=[]
                qstartF=[]
                qendR=[]
                qendF=[]
                qReads=0
                qFirstOnForward=-1
                qBad=False
            elif qBad: #same qname as previous ali but we know it's bad -> skip
                continue

            #################################################################################################
            #C] Either we're in the same qname and it's not bad, or we changed qname -> in both
            # cases update accumulators with current line
            if qname=="" :
                qname=align[0]

            # align[2] == chrom
            if qchrom=="":
                qchrom=align[2]
            elif qchrom!=align[2]:
                qBad=True
                dictStatCount["QNAliOnDiffChrSkip"]+=1
                continue
            # else same chrom, don't modify qchrom

            #Retrieving flags for STRAND and first/second read
            currentFlag=int(align[1])
            #flag 16 the alignment is on the reverse strand
            currentStrand="F"
            if currentFlag&16 :
                currentStrand="R"
            #flag 64 the alignment is first in pair, otherwise 128
            if currentFlag&64:
                qReads|=1
            else:
                qReads|=2

            #Processing for the positions lists for ali R and F
            currentStart=int(align[3])
            #Calculation of the CIGAR dependent 'end' position
            currentCigar=align[5]
            currentAliLength=ExtractAliLength(currentCigar)
            currentEnd=currentStart+currentAliLength-1
            if currentStrand=="F":
                qstartF.append(currentStart)
                qendF.append(currentEnd)
            else:
                qstartR.append(currentStart)
                qendR.append(currentEnd)

            # currentFirstOnForward==1 if according to this ali, the first-in-pair read is on
            # the forward strand, 0 otherwise
            currentFirstOnForward=0
            if ((currentFlag&64) and (currentStrand=="F")) or ((currentFlag&128) and (currentStrand=="R")) :
                currentFirstOnForward=1
            if qFirstOnForward==-1:
                # first ali for this qname
                qFirstOnForward=currentFirstOnForward
            elif qFirstOnForward!=currentFirstOnForward:
                qBad=True
                dictStatCount["QNSameReadDiffStrandSkip"]+=1
                continue
            # else this ali agrees with previous alis for qname -> NOOP

        #################################################################################################
        #VI]  Process last Qname
        Qname2ExonCount(qname,qchrom,qstartF,qendF,qstartR,qendR,qReads,dictIntervalTree,vecExonCount,dictStatCount)

        ##################################################################################################
        # VII] Check that the samtools commands have been carried out correctly
        p1.wait() #Waits for a child process to terminate. Changes the returncode attribute and returns it.
        try:
            p1.returncode == 0
        except ValueError as e:
            logger.error('the "Samtools sort" command encountered an error %s. error status : %s',e,p1.returncode)

        p2.wait()
        try:
            p2.returncode == 0
        except ValueError as e:
            logger.error('the "Samtools view" command encountered an error. error status : %s',e,p2.returncode)

        #################################################################################################
        #VIII] Process Monitoring
        NBTotalQname=(dictStatCount["QNAliOnDiffChrSkip"]+
                      dictStatCount["QNSameReadDiffStrandSkip"]+
                      dictStatCount["QNSingleReadSkip"]+
                      dictStatCount["QN1F&1R"]+
                      dictStatCount["QN2F&1R_2FOverlapSkip"]+
                      dictStatCount["QN2F&1R_SA"]+
                      dictStatCount["QN1F&2R_2ROverlapSkip"]+
                      dictStatCount["QN1F&2R_SA"]+
                      dictStatCount["QNAliGapLength>1000bpSkip"]+
                      dictStatCount["QNAliBackToBackSkip"]+
                      dictStatCount["QNAli2F&2R_SA"]+
                      dictStatCount["QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip"]+
                      dictStatCount["QNAliUnsuitableCombination_aliRorF>3Skip"])

        #################################################################################################
        #IX]Fragment count good progress control 
        #the last qname is not taken into account in the loop, hence the +1
        try:
            NBTotalQname==(dictStatCount["QNProcessed"])+1
            sum(vecExonCount)==dictStatCount["FragOverlapOnTargetInterval"]
            logger.info("CONTROL : all the qnames of %s were seen in the different conditions.",bamFile)
        except ValueError as e:
            statslist=dictStatCount.items()
            logger.error("Not all of %s's qnames were seen in the different conditions!!! Please check stats "+
                         "results below %s",bamFile,statslist)
            logger.error("Nb total Qname : %s. Nb Qname overlap target interval %s",NBTotalQname,sum(vecExonCount))
            vecExonCount=""

        #################################################################################################
        #X] Extract results
        return(vecExonCount)     
    

####################################################
# ExtractAliLength :
#This function retrieves the alignements length
#Input : a string in the CIGAR form
#Output : an int
def ExtractAliLength(CIGARAlign):
    length=0
    match = re.findall(r"(\d+)[MDN=X]",CIGARAlign)
    for Op in match:
        length+=int(Op)
    return(length)

####################################################
# Qname2ExonCount :
#This function identifies the fragments for each Qname
#When the fragment(s) are identified and it(they) overlap interest interval
# => increment count for this intervals.
#Inputs:
#   -chr variable [str]
#   -4 lists for R and F strand positions (start and end) [int]
#   -a bit variable informing about the read pair integrity 
#   (two bits : 01 First read was seen , 10 Second read was seen)
#   -the dictionary containing the interval trees for each chromosome
#   -list containing the fragment counts which will be completed/increment (final results)
#   -the dictionary to check the results at the process end.

#There is no output object, we just want to complete the fragment counts list by 
# interest intervals and complete the results control dictionary .

def Qname2ExonCount(chromString,startFList,endFList,startRList,endRList,readsBit,dictIntervalTree,vecExonCount,dictStatCount): # treatment on each Qname
    Frag=[] #fragment(s) intervals
    #####################################################################
    ###I) DIFFERENTS CONDITIONS ESTABLISHMENT : for fragment detection
    #####################################################################
    ########################
    #-Removing Qnames containing only one read (possibly split into several alignments, mostly due to MAPQ filtering)
    if readsBit!=3:
        dictStatCount["QNSingleReadSkip"]+=1
        return

    elif (1<len(startFList+startRList)<=3): # only 1 or 2 ali on each strand but not 2 on each
        dictStatCount["QNNbAliR&FBetween1-3"]+=1
        if max(startFList)<min(endRList):# alignments are not back-to-back
            GapLength=min(startRList)-max(endFList)# gap length between the two alignments (negative if overlapping)
            if (GapLength<=1000):
                if (len(startFList)==1) and (len(startRList)==1):# one ali on each strand, whether SA or not doesn't matter
                    dictStatCount["QN1F&1R"]+=1
                    Frag=[min(startFList[0],startRList[0]),max(endFList[0],endRList[0])]
                elif (len(startFList)==2) and (len(startRList)==1):
                    if (min(startFList)<max(endFList)) and (min(endFList)>max(startFList)):
                        dictStatCount["QN2F&1R_2FOverlapSkip"]+=1
                        return # Qname deletion if the ali on the same strand overlap (SV ?!)
                    else:
                        dictStatCount["QN2F&1R_SA"]+=1
                        Frag=[min(startFList[0],max(startRList)),max(endFList[0],max(endRList)),
                              min(startFList),min(endFList)]
                elif (len(startFList)==1) and (len(startRList)==2):
                    if (min(startRList)<max(endRList)) and (min(endRList)>max(startRList)):
                        dictStatCount["QN1F&2R_2ROverlapSkip"]+=1
                        return # Qname deletion if the ali on the same strand overlap (SV ?!)
                    else:
                        dictStatCount["QN1F&2R_SA"]+=1
                        Frag=[min(startFList[0],min(startRList)),max(endFList[0],min(endRList)),
                              max(startRList),max(endRList)]
            else: # Skipped Qname, GapLength too large, big del !?
                dictStatCount["QNAliGapLength>1000bpSkip"]+=1 # TODO extract the associated Qnames for breakpoint detection.
                return
        else: # Skipped Qname, ali back-to-back (SV, dup ?!)
            dictStatCount["QNAliBackToBackSkip"]+=1
            return

    elif(len(startFList)==2) and (len(startRList)==2): # only 2F and 2 R combinations
        #Only one possibility to generate fragments in this case
        if ( (min(startFList)<min(endRList)) and (min(endFList)>min(startRList)) and
             (max(startFList)<max(endRList)) and (max(endFList)>max(startRList)) ):
            Frag=[min(min(startFList),min(startRList)),
                   max(min(endFList),min(endRList)),
                   min(max(startFList),max(startRList)),
                   max(max(endFList),max(endRList))]
            dictStatCount["QNAli2F&2R_SA"]+=1
        else: # The alignments describe SVs that are too close => Skipped Qname
            dictStatCount["QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip"]+=1
            return
    else: # At least 3 alignments on one strand => Skipped Qname
        dictStatCount["QNAliUnsuitableCombination_aliRorF>3Skip"]+=1
        return
    #####################################################################
    ###II)- FRAGMENT COUNTING FOR EXONIC REGIONS
    #####################################################################
    #Retrieving the corresponding interval tree
    RefIntervalTree=dictIntervalTree[chromString]

    for idx in list(range(0,(int(len(Frag)/2)))):
        SelectInterval=RefIntervalTree.find_overlap(Frag[2*idx],Frag[2*idx+1])
        # how many overlapping intervals did we find
        overlaps=0
        for interval in SelectInterval:
            indexIntervalRef=int(interval[2])
            vecExonCount[indexIntervalRef]+=1
            dictStatCount["FragOverlapOnTargetInterval"]+=1
            overlaps += 1

        if overlaps==0:
            dictStatCount["QNNoOverlapOnTargetIntervalSkip"]+=1
            return #if the frag does not overlap a target interval => Skip
        else:
            dictStatCount["QNOverlapOnTargetInterval"]+=1

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
###################################
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

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","bams=","bams-from=","bed=","counts=","tmp=","threads="])
    except getopt.GetoptError as e:
        print("ERROR : "+e.msg+".\n",file=sys.stderr)  
        usage()
        sys.exit(1)

    for opt, value in opts:
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            usage()
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
            print("ERROR : Programming error. Unhandled option "+opt+".\n",file=sys.stderr)
            sys.exit(1)

    #####################################################
    # B) Checking that the mandatory parameters are presents
    if (bams=="" and bamsFrom=="") or (bams!="" and bamsFrom!=""):
        print("ERROR : You must use either --bams or --bams-from but not both.\n",file=sys.stderr)
        usage()
        sys.exit(1)
    if bedFile=="":
        print("ERROR : You must use --bedFile.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    #####################################################
    # C) Checking that the parameters actually exist and processing
    processBam=[]
    if bams!="":
        processBam=bams.split(",")
    
    if bamsFrom!="":
        if not os.path.isfile(bamsFrom):
            print("ERROR : BamFrom file "+bamsFrom+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr)
            sys.exit(1)
        else:
            bamListFile=open(bamsFrom,"r")
            for line in bamListFile:
                line = line.rstrip('\n')
                processBam.append(line)
    #Check all bam exist
    for b in processBam:
        if not os.path.isfile(b):
            print("ERROR : BAM "+b+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr)
            sys.exit(1)

    if (countFile!="") and (not os.path.isfile(countFile)):
        print("ERROR : Countfile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)

    if not os.path.isdir(tmpDir):
        print("ERROR : Tmp directory "+tmpDir+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr)
        sys.exit(1)

    if (threads<=0):
        print("ERROR : Threads number "+str(threads)+" is insufficient or negative.\n"+
        "Try "+scriptName+" --help.\n", file=sys.stderr) 
        sys.exit(1)  
    ######################################################
    # D) Parsing exonic intervals bed
    exonInterval=processBed(bedFile)

    ######################################################
    # E) Creating interval trees for each chromosome
    dictIntervalTree=intervalTreeDictCreation(exonInterval)

    ############################################
    # F) Parsing old counts file (.tsv) if exist else new count Dataframe creation 
    if os.path.isfile(countFile):
        counts=parseCountFile(countFile,exonInterval)
    else:
        counts=exonInterval

    #####################################################
    # G) Definition of a loop for each BAM files and the reads counting.
    for S in processBam:
        sampleName=os.path.basename(S)
        sampleName=sampleName.replace(".bam","")
        logger.info('Sample being processed : %s', sampleName)

        if sampleName in list(counts.columns[4:]):
            logger.info('Sample %s has already been analysed.', sampleName)
            continue
        else:
            FragVec=SampleCountingFrag(S,dictIntervalTree,exonInterval,tmpDir,threads)
            if FragVec!="":
                counts[sampleName]=FragVec
            else:
                logger.error("Sample %s encountered an error during the fragment counting,\n"+
                "which is excluded from the results file.\n", sampleName)
                continue

    counts.to_csv(sys.stdout,sep="\t", index=False, mode='a')
      

if __name__ =='__main__':
    main()
