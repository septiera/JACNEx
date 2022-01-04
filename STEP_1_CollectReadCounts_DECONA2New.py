#############################################################################################################
######################################## STEP1 Collect Read Count DECONA2 ###################################
#############################################################################################################
#This script allows you to perform a fragment count.
#This count is an alternative to the one performed by DECON because it generates erroneous fragment counts
#(e.g. split reads!!!)
#Our script allows to take into account all the particular cases of the reads paired association
#(thus it is only usable on paired-end technology) in the particular case of deletion.

#The input parameters are the following:
#-access path to the folder containing the bams to be processed
#-access path to the file containing the exonic intervals (bed file)
#-access path to the output folder (for storage)

#The output of this script is a tsv file in the same format as the interval file with the
#addition of the "FragCount" column for each sample processed.

#The first step of this script sorts the bams into Qname order and then filters the alignments
#according to specific flags (described below)
#This is done by the samtools Version: 1.9 (using htslib 1.9).
#The second step is to read the sam file line by line and count the fragments for each exonic
#interval (process details inside the script).

#############################################################################################################
################################ Loading of the modules required for processing #############################
#############################################################################################################
# 1) Python Modules
import sys #Path
import getopt
import logging
import os
import pandas as pd #read,make,treat Dataframe object
import numpy as np
import re  # regular expressions
import fnmatch
from ncls import NCLS # generating interval trees (cf : https://github.com/biocore-ntnu/ncls)
import subprocess #spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import io

# 2) Python Parallelization modules
import multiprocessing
from joblib import Parallel, delayed #parallelization characteristics works with the multiprocessing module.
num_cores=5 #CPU number definition to use during parallelization.
#Next Line define a CPUs minimum number to use for the process.
#Must be greater than the cores number defined below.
os.environ['NUMEXPR_NUM_THREADS'] = '10'

#####################################################################################################
################################ Logging Definition #################################################
#####################################################################################################
#create logger
logger = logging.getLogger(sys.argv[0])
logger.setLevel(logging.DEBUG)
#create console handler and set level to debug
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
#create formatter
formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)-8s [%(process)d] %(message)s', '%Y-%m-%d %H:%M:%S')
#add formatter to ch
ch.setFormatter(formatter)
#add ch to logger
logger.addHandler(ch)

#####################################################################################################
################################ Functions ##########################################################
#####################################################################################################
#Function allowing to test the existence of a file when only its path is indicated.
def checkFileExistance(filePath):
    try:
        with open(filePath, 'r') as f:
            logger.info('File %s exist', f)
    except FileNotFoundError as e:
        logger.error("File %s doesn't exist for this path. Please indicate a correct path.", filePath)
        sys.exit()
    except IOError as e:
        logger.error("File %s can't be opened for reading. Please check the files' formatting.", filePath)
        sys.exit()

###################################
#Function allowing to test the existence of a folder when only its path is indicated.
def checkFolderExistance(folderPath):
    if os.path.isdir(folderPath):
        logger.info('Folder %s exist', folderPath)
    else:
        logger.error("Folder %s doesn't exist for this path. Please indicate a correct path.", folderPath)
        sys.exit()

###################################
#Function allowing to create output folder.
def CreateAnalysisFolder(outFile, nameFolder):
    checkFolderExistance(outFile)
    nameFolder=os.path.join(outFile,nameFolder)
    try:
	    os.mkdir(nameFolder)
    except OSError:
	    logger.warning("Creation of the directory %s failed. The folder must already exist", nameFolder)
    else:
	    logger.info("Successfully created the directory %s ", nameFolder)
    return nameFolder

###################################
#Canonical transcripts sanity check.
#This function is useful for the bed file of canonical transcripts.
#It allows to check that the input file is clean.
#this function takes as input two arguments:
#-the path to the table under test.
#-the nature of this function verification+ extraction of the table ("TRUE") or simple syntax verification ("FALSE")
# It is divided into several steps:
#-Step 1: check that the loaded table contains 4 columns + name addition.
#-Step 2: Verification of column content
#   -- CHR : must start with "chr" and contain 25 different chromosomes (if less or more return warnings). The column must be in object format for processing.
#   -- START , END : must be in int64 format and not contain null value (necessary for comparison steps after calling CNVs)
#   -- TranscriptID_ExonNumber : all lines must start with ENST and end with an integer for exonNumber. The column must be in object format for processing.
#-Step 3: Allows you to indicate whether you want to work on the test table and therefore extract it from the function or not.
def BedSanityCheck(PathBedToCheck,TrueFalseExtractTab):
    BedToCheck=pd.read_table(PathBedToCheck,header=None,sep="\t")
    #####################
    #STEP1 : check that the loaded table contains 4 columns + columns renaming.
    #####################
    if len(BedToCheck.columns) == 4:
        logger.info("BedToCheck contains 4 columns.")
        BedToCheck.columns=["CHR","START","END","TranscriptID_ExonNumber"]
        #######################
        #STEP2 : Sanity Check
        #######################
        #CHR column
        if (len(BedToCheck[BedToCheck.CHR.str.startswith('chr')])==len(BedToCheck)) and (BedToCheck["CHR"].dtype=="O"):
            if len(BedToCheck["CHR"].unique())==25:
                logger.info("The CHR column has a correct format.")
            else:
                logger.warning("The canonical transcript file does not contain all the chromosomes = %s (Normally 24+chrM).", len(BedToCheck["CHR"].unique()))
        else:
            logger.error("The 'CHR' column doesn't have an adequate format. Please check it. The column must be a python object and each row must start with 'chr'.")
            sys.exit()
        #######################
        #Start and End column
        if (BedToCheck["START"].dtype=="int64") and (BedToCheck["END"].dtype=="int64"):
            logger.info("The 'START' 'END' columns have a correct format.")
            if (len(BedToCheck[BedToCheck.START<=0])>0) or (len(BedToCheck[BedToCheck.END<=0])>0):
                logger.error("Presence of outliers in the START and END columns. Values <=0.")
                sys.exit()
        else:
            logger.error("One or both of the 'START' and 'END' columns are not in the correct format. Please check. The columns must contain integers.")
            sys.exit()

        #######################
        #transcript_id_exon_number column
        if (len(BedToCheck[BedToCheck["TranscriptID_ExonNumber"].str.contains(r"^ENST.*_[0-9]{1,3}$")])==len(BedToCheck)) and (BedToCheck["CHR"].dtype=="O"):
            logger.info("The 'TranscriptID_ExonNumber' column has a correct format.")
        else:
            logger.error("The 'TranscriptID_ExonNumber' column doesn't have an adequate format. Please check it. The column must be a python object and each row must start with 'ENST'.")
            sys.exit()
    else:
        logger.error("BedToCheck doesn't contains 4 columns:"
                     +"\n CHR, START, END, TranscriptID_ExonNumber."
                     +"\n Please check the file before launching the script.")
        sys.exit()

    #######################
    #STEP3 : Extract TAB
    #######################
    if TrueFalseExtractTab=="TRUE":
        return(BedToCheck)
    elif TrueFalseExtractTab=="FALSE":
        logger.info("The function is used as sanity check no table extracted from this check.")
    else:
        logger.error("The argument indicating if we want to extract the table is wrong. Please indicate TRUE to extract the table or FALSE if the function is used as a check.")
        sys.exit()

###################################
#This function uses the ncls module to create interval trees
#The input : bed file with exon interval
#It produces a dictionary(hash): key=chr, values=interval_tree
def IntervalTreeDictCreation(BedIntervalFile):
    DictIntervalTree={}
    listCHR=list(BedIntervalFile.CHR.unique())
    for processChr in listCHR:
        intervalProcessChr=BedIntervalFile.loc[BedIntervalFile["CHR"]==processChr]
        ncls = NCLS(intervalProcessChr["START"].astype(int),intervalProcessChr["END"].astype(int),intervalProcessChr.index)
        DictIntervalTree[processChr]=ncls
    return(DictIntervalTree)

#########################################
#This function retrieves the alignements length
#The input is a string in the CIGAR form
#It returns an int
def ExtractAliLength(CIGARAlign):
    length=0
    match = re.findall(r"(\d+)[MDN=X]",CIGARAlign)
    for Op in match:
        length+=int(Op)
    return(length)

###################################
#This function identifies the fragments for each Qname
#When the fragment(s) are identified and it(they) overlap interest interval=> increment count for this intervals.
#The inputs are :
#-2 strings: qname and chr
#-4 lists for R and F strand positions (start and end)
#-the dictionary containing the interval trees for each chromosome
#-list containing the fragment counts which will be completed/increment
#-the dictionary to check the results at the process end.
#There is no output object, we just want to complete the fragment counts list by interest intervals.
def Qname2ExonCount(qnameString,chromString,startFList,endFList,startRList,endRList,ReadsBit,DictIntervalTree,VecExonCount,DictStatCount): # treatment on each Qname
    Frag=[] #fragment(s) intervals
    #####################################################################
    ###I) DIFFERENTS CONDITIONS ESTABLISHMENT : for fragment detection
    #####################################################################
    ########################
    #-Removing Qnames containing only one read (possibly split into several alignments, mostly due to MAPQ filtering)
    if ReadsBit!=3:
        DictStatCount["QNSingleReadSkip"]+=1
        return

    elif (1<len(startFList+startRList)<=3): # only 1 or 2 ali on each strand but not 2 on each
        DictStatCount["QNNbAliR&FBetween1-3"]+=1
        if len(startFList)==0:
            print(qnameString,chromString)
            print(startFList,endFList)
            print(startRList,endRList,ReadsBit)
            sys.exit()
        if max(startFList)<min(endRList):# alignments are not back-to-back
            GapLength=min(startRList)-max(endFList)# gap length between the two alignments (negative if overlapping)
            if (GapLength<=1000):
                if (len(startFList)==1) and (len(startRList)==1):# one ali on each strand, whether SA or not doesn't matter
                    DictStatCount["QN1F&1R"]+=1
                    Frag=[min(startFList[0],startRList[0]),max(endFList[0],endRList[0])]
                elif (len(startFList)==2) and (len(startRList)==1):
                    if (min(startFList)<max(endFList)) and (min(endFList)>max(startFList)):
                        DictStatCount["QN2F&1R_2FOverlapSkip"]+=1
                        return # Qname deletion if the ali on the same strand overlap (SV ?!)
                    else:
                        DictStatCount["QN2F&1R_SA"]+=1
                        Frag=[min(startFList[0],max(startRList)),max(endFList[0],max(endRList)),
                              min(startFList),min(endFList)]
                elif (len(startFList)==1) and (len(startRList)==2):
                    if (min(startRList)<max(endRList)) and (min(endRList)>max(startRList)):
                        DictStatCount["QN1F&2R_2ROverlapSkip"]+=1
                        return # Qname deletion if the ali on the same strand overlap (SV ?!)
                    else:
                        DictStatCount["QN1F&2R_SA"]+=1
                        Frag=[min(startFList[0],min(startRList)),max(endFList[0],min(endRList)),
                              max(startRList),max(endRList)]
            else: # Skipped Qname, GapLength too large, big del !?
                DictStatCount["QNAliGapLength>1000bpSkip"]+=1
                return
        else: # Skipped Qname, ali back-to-back (SV, dup ?!)
            DictStatCount["QNAliBackToBackSkip"]+=1
            return

    elif(len(startFList)==2) and (len(startRList)==2): # only 2F and 2 R combinations
        #Only one possibility to generate fragments in this case
        if ( (min(startFList)<min(endRList)) and (min(endFList)>min(startRList)) and
             (max(startFList)<max(endRList)) and (max(endFList)>max(startRList)) ):
            Frag=[min(min(startFList),min(startRList)),
                   max(min(endFList),min(endRList)),
                   min(max(startFList),max(startRList)),
                   max(max(endFList),max(endRList))]
            DictStatCount["QNAli2F&2R_SA"]+=1
        else: # The alignments describe SVs that are too close => Skipped Qname
            DictStatCount["QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip"]+=1
            return
    else: # At least 3 alignments on one strand => Skipped Qname
        DictStatCount["QNAliUnsuitableCombination_aliRorF>3Skip"]+=1
        return

    #####################################################################
    ###II)- FRAGMENT COUNTING FOR EXONIC REGIONS
    #####################################################################
    #Retrieving the corresponding interval tree
    RefIntervalTree=DictIntervalTree[chromString]

    for idx in list(range(0,(int(len(Frag)/2)))):
        SelectInterval=RefIntervalTree.find_overlap(Frag[2*idx],Frag[2*idx+1])
        # how many overlapping intervals did we find
        overlaps=0
        for interval in SelectInterval:
            indexIntervalRef=int(interval[2])
            VecExonCount[indexIntervalRef]+=1
            DictStatCount["FragOverlapOnTargetInterval"]+=1
            overlaps += 1

        if overlaps==0:
            DictStatCount["QNNoOverlapOnTargetIntervalSkip"]+=1
            return #if the frag does not overlap a target interval => Skip
        else:
            DictStatCount["QNOverlapOnTargetInterval"]+=1


##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
###################################
def main(argv):
    ##########################################
    # A) ARGV parameters definition
    intervalFile = ''
    bamOrigFolder=''
    outputFile =''

    try:
        opts, args = getopt.getopt(argv,"h:i:b:o:",["help","intervalFile=","bamfolder=","outputfile="])
    except getopt.GetoptError:
        print('python3.6 STEP_1_CollectReadCounts_DECONA2New.py -i <intervalFile> -b <bamfolder> -o <outputfile>')
        sys.exit(2)
    for opt, value in opts:
        if opt == '-h':
            print("COMMAND SUMMARY:"
            +"\n This script performs READ counts using the bedtools software from exome sequencing data."
            +"\n"
            +"\n USAGE:"
            +"\n python3.6 STEP_1_CollectReadCounts_DECONA2New.py -i <intervalfile> -b <bamfolder> -o <outputfile>"
            +"\n"
            +"\n OPTIONS:"
            +"\n	-i : A bed file obtained in STEP0. Please indicate the full path.(4 columns : CHR, START, END, TranscriptID_ExonNumber)"
            +"\n	-b : The path to the folder containing the BAM files."
            +"\n	-o : The path where create the new output file for the current analysis.")
            sys.exit()
        elif opt in ("-i", "--intervalFile"):
            intervalFile=value
        elif opt in ("-b", "--bamfolder"):
            bamOrigFolder=value
        elif opt in ("-o", "--outputfile"):
            outputFile=value

    #Check that all the arguments are present.
    logger.info('Intervals bed file path is %s', intervalFile)
    logger.info('BAM folder path is %s', bamOrigFolder)
    logger.info('Output file path is %s ', outputFile)

    #####################################################
    # A) Setting up the analysis tree.
        # 1) Creation of the DataToProcess folder if not existing.
    DataToProcessPath=CreateAnalysisFolder(outputFile,"DataToProcess")
        # 2) Creation of the ReadCount folder if not existing.
    RCPath=CreateAnalysisFolder(DataToProcessPath,"ReadCount")
        # 3) Creation of the Bedtools folder if not existing.
    RCPathOutput=CreateAnalysisFolder(RCPath,"DECONA2")

    #####################################################
    # B)Bam files list extraction
    #This list contains the path to each bam.
    checkFolderExistance(bamOrigFolder)
    sample=fnmatch.filter(os.listdir(bamOrigFolder), '*.bam')
    inputs=np.arange(0,len(sample))

    #####################################################
    # C) Canonical transcript bed, existence check
    checkFileExistance(intervalFile)
    # Bed file load and Sanity Check : If the file does not have the expected format, the process ends.
    intervalBed=BedSanityCheck(intervalFile, "TRUE") #The table will not be used but the verification of its format is necessary.

    ############################################
    # D]  Creating interval trees for each chromosome
    DictIntervalTree=IntervalTreeDictCreation(intervalBed)

    #####################################################
    # E) Definition of a loop for each sample allowing the BAM files symlinks and the reads counting.
    def SampleParalelization(i):
        bamFile=sample[i]
        sampleName=bamFile.replace(".bam","")
        bamFile=os.path.join(bamOrigFolder,bamFile)
        logger.info('Sample being processed : ', sampleName)

        ############################################
        #I] Check that the process has not already been applied to the current bam.
        NameCountFile=os.path.join(RCPathOutput,sampleName+".tsv")
        if (os.path.isfile(NameCountFile)) and (sum(1 for _ in open(NameCountFile))==len(intervalBed)):
            logger.info("File ",NameCountFile," already exists and has the same lines number as the bed file ", intervalFile)
            return # Bam re-analysis is not effective.
        #else : We process the bam without tsv file or with incomplete tsv file.
        if os.path.isfile(NameCountFile): #Be careful when updating the bed if there is no change in the output directory, all the files are replaced.
            logger.warning("File ",NameCountFile," already exists but it's truncated. This involves replacing the corrupted file.")
            CommandLine="rm "+NameCountFile
            returned_value=os.system(CommandLine)
            #Security Check
            if returned_value == 0:
                print("File deletion has been carried out correctly.")
            else:
                print("File deletion has encountered an error ", CommandLine)
                sys.exit()
        #else : processing can then be done on the current bam.

        ############################################
        # II] Pretreatments on the sample bam.
        # Samtools commands line :
        # The bam should be sorted by Qname to facilitate the fragments counting (-n option).
        # -@ allows the process to be split across multiple cores.
        # -T is used to store the process temporary files in the ram memory.
        # -O indicates the stdout result must be in sam format.
        cmd1 ="samtools sort -n "+bamFile+" -@ "+str(num_cores)+" -T /mnt/RamDisk/ -O sam"

        # Using samtools view, the following flag filters can be performed with the -F option:
        #   -4 0x4 Read Unmapped
        #   -256 0x80 not primary alignment
        #   -512 0x200 Read fails platform/vendor quality checks
        #   -1024 0x400 Read is PCR or optical duplicate
        # Total Flag decimal = 1796
        # The command also integrates a filtering on the mapping quality here fixed at 20 (-q option).
        # By default the stdout is in sam format
        cmd2 ="samtools view -F 1796 -q 20 -@ "+str(num_cores)
        # Orders execution
        p1 = subprocess.Popen(cmd1.split(),stdout=subprocess.PIPE,bufsize=1)
        p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout,stdout=subprocess.PIPE,bufsize=1,encoding='utf8')

        # III] Outpout variables initialization
        #list containing the fragment counts for the exons.
        VecExonCount=[0]*len(intervalBed) #The indexes correspond to the exon order in the "intervalBed".

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
        DictStatCount={ i : 0 for i in keys }

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
            DictStatCount["AlignmentsInBam"]+=1

            #################################################################################################
            #A] Selection of information for the current alignment only if it is on main chr
            if not mainChr.match(align[2]): continue #TODO make this work for different naming conventions
            DictStatCount["AlignmentsOnMainChr"]+=1 #Only alignments on the main chromosomes are treated

            #################################################################################################
            #B] If we are done with previous qname: process it and reset accumulators
            if (qname!=align[0]) and (qname!=""):  # align[0] is the qname
                DictStatCount["QNProcessed"]+=1
                if not qBad:
                    Qname2ExonCount(qname,qchrom,qstartF,qendF,qstartR,qendR,qReads,DictIntervalTree,VecExonCount,DictStatCount)
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
            #c] Either we're in the same qname and it's not bad, or we changed qname -> in both
            # cases update accumulators with current line
            if qname=="" :
                qname=align[0]

            # align[2] == chrom
            if qchrom=="":
                qchrom=align[2]
            elif qchrom!=align[2]:
                qBad=True
                DictStatCount["QNAliOnDiffChrSkip"]+=1
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
                DictStatCount["QNSameReadDiffStrandSkip"]+=1
                continue
            # else this ali agrees with previous alis for qname -> NOOP


        #################################################################################################
        #VI]  Process last Qname
        Qname2ExonCount(qname,qchrom,qstartF,qendF,qstartR,qendR,qReads,DictIntervalTree,VecExonCount,DictStatCount)

        #################################################################################################
        #VII] Process Monitoring
        NBTotalQname=(DictStatCount["QN1F&1R_SA&notSA"]+
                DictStatCount["QNAliOnDiffChrSkip"]+
                DictStatCount["QNAliSameStrandOrAloneSkip"]+
                DictStatCount["QN2F&1R_2FOverlapSkip"]+
                DictStatCount["QN2F&1R_SA"]+
                DictStatCount["QN1F&2R_2ROverlapSkip"]+
                DictStatCount["QN1F&2R_SA"]+
                DictStatCount["QNAliGapLength>1000bpSkip"]+
                DictStatCount["QNAliBackToBackSkip"]+
                DictStatCount["QNAli2F&2R_SA"]+
                DictStatCount["QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip"]+
                DictStatCount["QNAliUnsuitableCombination_aliRorF>3Skip"]+
                DictStatCount["QNNoOverlapFragOnTargetIntervalSkip"])
        #the last qname is not taken into account in the loop, hence the +1
        if NBTotalQname==DictStatCount["QNProcessed"]+1 and sum(VecExonCount)==DictStatCount["QNOverlapFragOnTargetInterval"]:
            logger.info("CONTROL : all the qnames of "+bamFile+" were seen in the different conditions.")
        else:
            logger.error("Not all of "+bamFile+"'s qnames were seen in the different conditions!!! Please check results below "+DictStatCount)
            sys.exit()
        #################################################################################################
        #IX] Saving the tsv file
        SampleTsv=intervalBed
        SampleTsv['FragCount']=VecExonCount
        SampleTsv.to_csv(os.path.join(NameCountFile),sep="\t", index=False, header=None)

    Parallel(n_jobs=num_cores,backend="threading")(delayed(SampleParalelization)(i) for i in inputs)# TODO find an alternative to backend="threading"

if __name__ =='__main__':
    main(sys.argv[1:])
