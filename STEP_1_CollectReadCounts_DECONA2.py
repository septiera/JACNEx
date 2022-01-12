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
#-access path to tmpDir

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
import sys
import getopt
import logging
import os
import pandas as pd #read,make,treat Dataframe object
import re  # regular expressions
import fnmatch
from ncls import NCLS # generating interval trees (cf : https://github.com/biocore-ntnu/ncls)
import subprocess #spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import io
import tempfile #manages the creation and deletion of temporary folders/files

# 2) Python Parallelization modules
#import multiprocessing
#from joblib import Parallel, delayed #parallelization characteristics works with the multiprocessing module.
#num_cores=20 #CPU number definition to use during parallelization.
#Next Line define a CPUs minimum number to use for the process.
#Must be greater than the cores number defined below.
#os.environ['NUMEXPR_NUM_THREADS'] = '10'

#####################################################################################################
################################ Logging Definition #################################################
#####################################################################################################
#create logger : Loggers expose the interface that the application code uses directly
logger=logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.DEBUG)
#create console handler and set level to debug : The handlers send the log entries (created by
#the loggers) to the desired destinations.
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
#create formatter : Formatters specify the structure of the log entry in the final output.
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')
#add formatter to ch(handler)
ch.setFormatter(formatter)
#add ch(handler) to logger
logger.addHandler(ch)

#####################################################################################################
################################ Functions ##########################################################
#####################################################################################################
####################################################
#Function allowing to create output folder.
#Input : take output path
#create outdir if it doesn't exist, die on failure
def CreateFolder(outdir):
    if not os.path.isdir(outdir):
        try:
            os.mkdir(outdir)
        except OSError as error:
            logger.error("Creation of the directory %s failed : %s", outdir, error.strerror)
            sys.exit()
  
####################################################
#Canonical transcripts sanity check.
#This function is useful for the bed file of canonical transcripts.
#It allows to check that the input file is clean.

#Input:
#-the path to the table under test.
# It is divided into several steps:
#-Step I): check that the loaded table contains 4 columns + name addition.
#-Step II): Column content verification
#   -- CHR : must start with "chr" and contain 25 different chromosomes (if less or more return warnings).
#The column must be in object format for processing.
#   -- START , END : must be in int64 format and not contain null value (necessary for comparison steps
#after calling CNVs)
#   -- TranscriptID_ExonNumber : all lines must start with ENST and end with an integer for exonNumber.
#The column must be in object format for processing.

#Output: This function returns a dataframe 

def BedParseAndSanityCheck(PathBedToCheck):
    if not os.path.isfile(PathBedToCheck):
        logger.error("file %s doesn't exist ",PathBedToCheck)
        sys.exit()
    BedToCheck=pd.read_table(PathBedToCheck,header=None,sep="\t")
    #####################
    #I): check that the loaded table contains 4 columns + columns renaming.
    #####################
    if len(BedToCheck.columns) == 4:
        logger.info("BedToCheck contains 4 columns.")
        BedToCheck.columns=["CHR","START","END","TranscriptID_ExonNumber"]
        #######################
        #II) : Sanity Check
        #######################
        #CHR column
        if (len(BedToCheck[BedToCheck.CHR.str.startswith('chr')])==len(BedToCheck)) and (BedToCheck["CHR"].dtype=="O"):
            if len(BedToCheck["CHR"].unique())==25:
                logger.info("The CHR column has a correct format.")
            else:
                logger.warning("The canonical transcript file does not contain all the chromosomes = %s "+
                               "(Normally 24+chrM).", len(BedToCheck["CHR"].unique()))
        else:
            logger.error("The 'CHR' column doesn't have an adequate format. Please check it. The column "+
                         "must be a python object and each row must start with 'chr'.")
            sys.exit()
        #######################
        #Start and End column
        if (BedToCheck["START"].dtype=="int64") and (BedToCheck["END"].dtype=="int64"):
            logger.info("The 'START' 'END' columns have a correct format.")
            if (len(BedToCheck[BedToCheck.START<=0])>0) or (len(BedToCheck[BedToCheck.END<=0])>0):
                logger.error("Presence of outliers in the START and END columns. Values <=0.")
                sys.exit()
        else:
            logger.error("One or both of the 'START' and 'END' columns are not in the correct format. "+
                         "Please check. The columns must contain integers.")
            sys.exit()

        #######################
        #transcript_id_exon_number column
        if (len(BedToCheck[BedToCheck["TranscriptID_ExonNumber"].str.contains(r"^ENST.*_[0-9]{1,3}$")])==len(BedToCheck)
            and BedToCheck["CHR"].dtype=="O"):
            logger.info("The 'TranscriptID_ExonNumber' column has a correct format.")
        else:
            logger.error("The 'TranscriptID_ExonNumber' column doesn't have an adequate format. Please "+
                         "check it. The column must be a python object and each row must start with 'ENST'.")
            sys.exit()
    else:
        logger.error("BedToCheck doesn't contains 4 columns:"
                     +"\n CHR, START, END, TranscriptID_ExonNumber."
                     +"\n Please check the file before launching the script.")
        sys.exit()

    return(BedToCheck)
    

####################################################
#This function uses the ncls module to create interval trees
#Input : bed file with exon interval
#Output: dictionary(hash): key=chr, value=interval_tree
def IntervalTreeDictCreation(BedIntervalFile):
    DictIntervalTree={}
    listCHR=list(BedIntervalFile.CHR.unique())
    for processChr in listCHR:
        intervalProcessChr=BedIntervalFile.loc[BedIntervalFile["CHR"]==processChr]
        ncls = NCLS(intervalProcessChr["START"].astype(int),intervalProcessChr["END"].astype(int),intervalProcessChr.index)
        DictIntervalTree[processChr]=ncls
    return(DictIntervalTree)

####################################################
#This function retrieves the alignements length
#The input is a string in the CIGAR form
#Output : an int
def ExtractAliLength(CIGARAlign):
    length=0
    match = re.findall(r"(\d+)[MDN=X]",CIGARAlign)
    for Op in match:
        length+=int(Op)
    return(length)

####################################################
#This function checks the presence of a results file for the sample being processed and performs an integrity check.
#If the file does not exist or the integrity of the old file is not validated, the count will be performed.
#Otherwise the process will be skipped for that sample.
#Inputs: 
#- access path to the tsv
#- panda dataframe containing all genomic exons positions 
#Output: a Boolean variable

def SanityCheckPreExistingTSV(countFilePath,BedIntervalFile):
    validSanity=True #Boolean: 0 analysis/reanalysis; 1 no analysis
    numberExons=len(BedIntervalFile)
    if (os.path.isfile(countFilePath)) and (sum(1 for _ in open(countFilePath))==numberExons):
        #file exists and has the same lines number as the exonic interval file
        logger.info("File %s already exists and has the same lines number as the bed file %s ",countFilePath,numberExons)
        count=pd.read_table(countFilePath,header=None,sep="\t")
        #Columns comparisons
        if (count[0].equals(BedIntervalFile['CHR']) and count[1].equals(BedIntervalFile['START']) and
            count[2].equals(BedIntervalFile['END']) and count[3].equals(BedIntervalFile['TranscriptID_ExonNumber'])):
            validSanity=False# Bam re-analysis is not effective.
        else:
            logger.warning("However there are differences between the columns. Check and correct these differences.",
                           countFilePath)
            try:
                os.remove(countFilePath)
            except OSError as error:
                logger.error("File deletion has encountered an error %s %s", countFilePath, error.strerror)
                sys.exit()
            
    elif os.path.isfile(countFilePath):
        #Be careful when updating the bed if there is no change in the output directory, all the files are replaced.
        logger.warning("File %s already exists but it's truncated. This involves replacing the corrupted file.",
                       countFilePath)
        try:
            os.remove(countFilePath)
        except OSError as error:
            logger.error("File deletion has encountered an error %s %s", countFilePath, error.strerror)
            sys.exit()
    
    return(validSanity)

####################################################
#This function identifies the fragments for each Qname
#When the fragment(s) are identified and it(they) overlap interest interval=> increment count for this intervals.

#Inputs:
#-2 strings: qname and chr
#-4 lists for R and F strand positions (start and end)
#-a bit variable informing about the read pair integrity (two bits : 01 First read was seen , 10 Second read was seen)
#-the dictionary containing the interval trees for each chromosome
#-list containing the fragment counts which will be completed/increment (final results)
#-the dictionary to check the results at the process end.

#There is no output object, we just want to complete the fragment counts list by interest intervals.

def Qname2ExonCount(qnameString,chromString,startFList,endFList,startRList,endRList,readsBit,dictIntervalTree,
                    vecExonCount,dictStatCount): # treatment on each Qname
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
        if len(startFList)==0:
            print(qnameString,chromString)
            print(startFList,endFList)
            print(startRList,endRList,readsBit)
            sys.exit()
        if max(startFList)<min(endRList):# alignments are not back-to-back
            GapLength=min(startRList)-max(endFList)# gap length between the two alignments (negative if overlapping)
            if (GapLength<=1000):
                if (len(startFList)==1) and (len(startRList)==1):
                    # one ali on each strand, whether SA or not doesn't matter
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
                dictStatCount["QNAliGapLength>1000bpSkip"]+=1
                # TODO extract the associated Qnames for breakpoint detection.
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


####################################################
#This function allows samples to be processed for fragment counting.
#It uses several other functions such as: ExtractAliLength and Qname2ExonCount.
#Inputs:
#-the full path to the bam file
#-the path with the name of the output file (fragment count file)
#-the dictionary of exonic interval trees 
#-a dataframe of the exonic intervals (for output)
#-the path to the temporary file
#-the number of cpu allocated for processing (used for samtools commands)

#Temporary variables created:
#It uses samtools commands to filter reads and sort them in Qnames order (.sam file generated in a temporary file).
#It creates a list of fragment count results for each line index of the bed file.
#It also creates a dictionary containing all the counts for each condition imposed to obtain a fragment count. 
#This dictionary is used as a control in order not to miss any condition not envisaged (number of Qname
#processed must be the same at the end of the process).

#Output: tsv files formatted as follows (no row or column index, 5 columns => CHR, START, END, ENSTID_Exon, FragCount)

def SampleCountingFrag(bamFile,nameCountFilePath,dictIntervalTree,intervalBed,processTmpDir,num_cores):
    with tempfile.TemporaryDirectory(dir=processTmpDir) as SampleTmpDir:
        numberExons=len(intervalBed)
        logger.info('Sample BAM being processed : %s', bamFile)

        ############################################
        # I] Pretreatments on the sample bam.
        # Samtools commands line :
        # The bam should be sorted by Qname to facilitate the fragments counting (-n option).
        # -@ allows the process to be split across multiple cores.
        # -T is used to store the process temporary files in the ram memory.
        # -O indicates the stdout result must be in sam format.
        cmd1 ="samtools sort -n "+bamFile+" -@ "+str(num_cores)+" -T "+SampleTmpDir+" -O sam"

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

        #################################################################################################
        #VII] Process Monitoring
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

        #IX]Fragment count good progress control 
        #the last qname is not taken into account in the loop, hence the +1
        if ((NBTotalQname==(dictStatCount["QNProcessed"])+1) and
            (sum(vecExonCount)==dictStatCount["FragOverlapOnTargetInterval"])):
            logger.info("CONTROL : all the qnames of %s were seen in the different conditions.",bamFile)
        else:
            statslist=dictStatCount.items()
            logger.error("Not all of %s's qnames were seen in the different conditions!!! Please check stats "+
                         "results below %s",bamFile,statslist)
            logger.error("Nb total Qname : %s. Nb Qname overlap target interval %s",NBTotalQname,sum(vecExonCount))
            sys.exit()
        #################################################################################################
        #X] Saving the tsv file
        SampleTsv=intervalBed
        SampleTsv['FragCount']=vecExonCount
        SampleTsv.to_csv(os.path.join(nameCountFilePath),sep="\t", index=False, header=None)


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
    processTmpDir=''

    try:
        opts, args = getopt.getopt(argv,"h:i:b:o:t:",["help","intervalFile=","bamfolder=","outputfile=","tmpfolder="])
    except getopt.GetoptError:
        print(sys.argv[0]+' -i <intervalFile> -b <bamfolder> -o <outputfile> -t <tmpfolder>')
        sys.exit(2)
    for opt, value in opts:
        if opt == '-h':
            print("COMMAND SUMMARY:"
            +"\n This script performs READ counts using the bedtools software from exome sequencing data."
            +"\n"
            +"\n USAGE:"
            +"\n"+sys.argv[0]+" -i <intervalfile> -b <bamfolder> -o <outputfile> -t <tmpfolder>"
            +"\n"
            +"\n OPTIONS:"
            +"\n	-i : A bed file obtained in STEP0. Please indicate the full path.(4 columns :"+
            +"\n             CHR, START, END, TranscriptID_ExonNumber)"
            +"\n	-b : The path to the folder containing the BAM files."
            +"\n	-o : The path where create the new output file for the current analysis."
            +"\n	-t : The path to the temporary folder containing the samtools results ")
            sys.exit()
        elif opt in ("-i", "--intervalFile"):
            intervalFile=value
        elif opt in ("-b", "--bamfolder"):
            bamOrigFolder=value
        elif opt in ("-o", "--outputfile"):
            outputFile=value
        elif opt in ("-t","--tmpfolder"):
            processTmpDir=value
            
    #Check that all the arguments are present.
    logger.info('Intervals bed file path is %s', intervalFile)
    logger.info('BAM folder path is %s', bamOrigFolder)
    logger.info('Output file path is %s ', outputFile)
    logger.info('Temporary folder path is %s ', processTmpDir)

    #####################################################
    # A) Setting up the analysis tree.
        # 1) Creation of the DataToProcess folder if not existing.
    DataToProcessPath=outputFile+"DataToProcess"
    CreateFolder(DataToProcessPath)
        # 2) Creation of the ReadCount folder if not existing.
    RCPath=DataToProcessPath+"/ReadCount"
    CreateFolder(RCPath)
        # 3) Creation of the Bedtools folder if not existing.
    RCPathOutput=RCPath+"/DECONA2"
    CreateFolder(RCPathOutput)

    #####################################################
    # B)Bam files list extraction
    #This list contains the path to each bam.
    if not os.path.isdir(bamOrigFolder):
        logger.error("Bam folder doesn't exist %s",bamOrigFolder)
        sys.exit()
    
    samples=fnmatch.filter(os.listdir(bamOrigFolder), '*.bam')

    #####################################################
    # C) Canonical transcript bed, existence check    
    # Bed file load and Sanity Check : If the file does not have the expected format, the process ends.
    intervalBed=BedParseAndSanityCheck(intervalFile)

    ############################################
    # D]  Creating interval trees for each chromosome
    dictIntervalTree=IntervalTreeDictCreation(intervalBed)

    #####################################################
    # E) Definition of a loop for each sample allowing the BAM files symlinks and the reads counting.
    for S in samples:
        sampleName=S.replace(".bam","")
        bamFile=os.path.join(bamOrigFolder,S)
        logger.info('Sample being processed : %s', sampleName)

        #output TSV name definition
        nameCountFilePath=os.path.join(RCPathOutput,sampleName+".tsv")
        validSanity=SanityCheckPreExistingTSV(nameCountFilePath,intervalBed)
        print(sampleName," : never seen ",validSanity)

        if validSanity:
            SampleCountingFrag(bamFile,nameCountFilePath,dictIntervalTree,intervalBed,processTmpDir,num_cores)
        else:
            continue    

if __name__ =='__main__':
    main(sys.argv[1:])
