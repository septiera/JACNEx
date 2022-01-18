#############################################################################################################
######################################## STEP1 Collect Read Count DECONA2 ###################################
#############################################################################################################

#############################################################################################################
################################ Loading of the modules required for processing #############################
#############################################################################################################
# 1) Python Modules
import sys #Path
import getopt
import logging
import os
from numpy.lib.shape_base import tile
import pandas as pd #read,make,treat Dataframe object
import re  # regular expressions
import fnmatch
from ncls import NCLS # generating interval trees (cf : https://github.com/biocore-ntnu/ncls)
import subprocess #spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import tempfile #manages the creation and deletion of temporary folders/files

#Definition of the scripts'execution date (allows to annotate the output files to track them over time) 
import time
now=time.strftime("%y%m%d")

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
    print("COMMAND SUMMARY:\n"+
"Fragments counting for exonic intervals(.bed) using paired-end alignment file(.bam):\n"+
"   1.Exonic intervals file parsing and preparing.\n"+
"   2.Interval trees initialisation (with ncls python module).\n"+
"   3.New Patients identification for analysis.\n"+
"   4.Sorting .bam by QNAME (with samtools sort).\n"+
"   5.Filtering tmpsort.sam (with samtools view).\n"+
"   6.Fragment counting and results sanity check.\n"+
"Script will print to stdout a new countFile in TSV format, copying the data from the pre-existing\n"+
"countFile if provided and adding columns with counts for the new BAMs/samples..\n\n"
"OPTIONS:\n"+
"   -i or --bam [str]: a bam file or a bam list (with path)\n"+
"   -b or --bed [str]: a bed file (with path), possibly gzipped, containing exon definitions \n"+
"                               formatted as CHR START END EXON_ID\n"+
"   -c or --counts [str] optional: a pre-parsed count file (with path), old fragment count file  \n"+
"                                        to be completed if new patient(s) is(are) added. \n"
"   -t or --tmp [str] optional: a temporary folder (with path),allows to save temporary files \n"+
"                                     from samtools sort. By default, this is placed in '/tmp'.\n"+
"                                     Tip: place the storage preferably in RAM.\n"+
"   -j or --jobs [int] optional: number of threads to allocate for samtools. By default, the value is set to 10.\n")


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
        logger.error("Exon intervals file %s doesn't exist.",bedname)
        sys.exit()
    BedToCheck=pd.read_table(PathBedToCheck,header=None,sep="\t")
    #####################
    #I): Sanity Check
    #####################
    if len(BedToCheck.columns) == 4:
        logger.info("%s contains 4 columns.",bedname)
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
            logger.error("The 'CHR' column doesn't have an adequate format. Please check it.\n"
                        +"The column must be a python object [str]")
            sys.exit()
        #######################
        #Start and End column
        if (BedToCheck["START"].dtype=="int64") and (BedToCheck["END"].dtype=="int64"):
            if (len(BedToCheck[BedToCheck.START<=0])>0) or (len(BedToCheck[BedToCheck.END<=0])>0):
                logger.error("Presence of outliers in the START and END columns. Values <=0.")
                sys.exit()
        else:
            logger.error("One or both of the 'START' and 'END' columns are not in the correct format. Please check it.\n"
                        +"The columns must contain integers.")
            sys.exit()
        #######################
        #transcript_id_exon_number column
        if not (BedToCheck["EXON_ID"].dtype=="O"):
            logger.error("The 'EXON_ID' column doesn't have an adequate format. Please check it.\n"
                        +"The column must be a python object.")
            sys.exit()
    else:
        logger.error("BedToCheck doesn't contains 4 columns:\n"
                     +"CHR, START, END, EXON_ID.\n"
                     +"Please check the file before launching the script.")
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
def IntervalTreeDictCreation(BedIntervalFile):
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
# - countFile is a tsv file (with path), including column titles, as
#   specified previously
# - exons is a dataframe holding exon definitions, padded and sorted,
#   as returned by processBed
#
# -> Parse countFile into a dataframe (will be returned)
# -> Check that the first 4 columns are identical to exons,
#    otherwise die with an error.
# -> check that the samples counts columns are not empty(>15000000 frag counts)
# 
# Output:
# - returns the Frag count results as a pandas dataframe with column headers CHR START END EXON_ID,sampleName(n)
# - list of samples to be reanalysed
# - list of samples with correct count (to be deleted from the process).
def parseCountFile(countFile, exons):
    try:
        counts=pd.read_table(countFile,sep="\t")
    except Exception as e:
        logger.error("Parsing provided countFile %s: %s", countFile, e)
        sys.exit(1)

    if (os.path.isfile(counts)) and (len(counts)==len(exons)): #Sanity Check
        #Columns comparisons
        if not ((counts[0].equals(exons['CHR'])) and 
                (counts[1].equals(exons['START'])) and 
                (counts[2].equals(exons['END'])) and 
                (counts[3].equals(exons['EXON_ID']))):
            logger.error("Old counts file %s does not have the same columns as the exonic interval file.\n"+
            "Transcriptome version has probably changed.\n"+
            "In this case the whole samples set re-analysis must be done.\n"+
            "Do not set the -p option.",countFile)
    return(counts)

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
#This function identifies the fragments for each Qname
#When the fragment(s) are identified and it(they) overlap interest interval=> increment count for this intervals.

#Inputs:
#- qname and chr variable [str]
#-4 lists for R and F strand positions (start and end) [int]
#-a bit variable informing about the read pair integrity (two bits : 01 First read was seen , 10 Second read was seen)
#-the dictionary containing the interval trees for each chromosome
#-list containing the fragment counts which will be completed/increment (final results)
#-the dictionary to check the results at the process end.

#There is no output object, we just want to complete the fragment counts list by interest intervals.

def Qname2ExonCount(qnameString,chromString,startFList,endFList,startRList,endRList,readsBit,dictIntervalTree,vecExonCount,dictStatCount): # treatment on each Qname
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
            logger.error(qnameString,chromString,"\n",startFList,endFList,"\n",startRList,endRList,readsBit)
            sys.exit()
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
#This dictionary is used as a control in order not to miss any condition not envisaged (number of Qname processed must be the same at the end of the process).

#Output: tsv files formatted as follows (no row or column index, 5 columns => CHR, START, END, ENSTID_Exon, FragCount)

def SampleCountingFrag(bamFile,sampleName,dictIntervalTree,intervalBed,processTmpDir,num_cores):
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
                    Qname2ExonCount(qname,qchrom,qstartF,qendF,qstartR,qendR,qReads,dictIntervalTree,vecExonCount,dictStatCount)
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
        if (NBTotalQname==(dictStatCount["QNProcessed"])+1) and (sum(vecExonCount)==dictStatCount["FragOverlapOnTargetInterval"]):
            logger.info("CONTROL : all the qnames of %s were seen in the different conditions.",bamFile)
        else:
            statslist=dictStatCount.items()
            logger.error("Not all of %s's qnames were seen in the different conditions!!! Please check stats results below %s",bamFile,statslist)
            logger.error("Nb total Qname : %s. Nb Qname overlap target interval %s",NBTotalQname,sum(vecExonCount))
            sys.exit()
        #################################################################################################
        #X] Saving the tsv file
        if sampleName in counts.columns():

        SampleTsv=intervalBed
        SampleTsv['FragCount']=vecExonCount
        SampleTsv.to_csv(os.path.join(sampleName),sep="\t", index=False, header=None)

        p1.communicate() #allows to wait for the process end to return the following status error
        if p1.returncode != 0:
            logger.error('the "Samtools sort" command encountered an error. error status : %s',p1.returncode)
            sys.exit()
        p2.terminate()

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
###################################
def main():
    ##########################################
    # A) ARGV parameters definition
    # default setting ARGV 
    cpu=10 
    processTmpDirPath="/tmp"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'hi:b:c:t:j:',
        ["help","bam=","bed=","counts=","tmp=","jobs="])
        if not opts:
            sys.stderr.write("ERROR : No options supplied. Please follow the instructions.\n\n")
            usage()
            sys.exit()   
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : "+e.msg+". Please follow the instructions.\n\n")  
        usage()
        sys.exit()

    for opt, value in opts:
            #variables association with user parameters (ARGV)
            if opt in ('-h', '--help'):
                usage()
                sys.exit(2)
            elif opt in ("-i", "--bam"):
                bamFilePath=value
            elif opt in ("-b", "--bed"):
                bedFilePath =value
            elif opt in ("-c","--counts"):
                countFilePath=value
            elif opt in ("-t","--tmp"):
                tmpDirPath=value
            elif opt in ("-j","--jobs"):
                cpu=value

    #Check that all the arguments are present.
    logger.debug("BAM folder path is %s", bamFilePath)
    logger.debug("Intervals bed file path is %s", bedFilePath)
    logger.debug("Old fragment count file path is %s", countFilePath)
    logger.debug("Temporary folder path is %s", tmpDirPath)
    logger.debug("CPU number used is %s", cpu)

    #####################################################
    # A) Output folder creation
    ####################################################
    outputFolder="DECONA2_FragCountResults_"+now
    try:
        os.mkdir(outputFolder)
    except OSError as error:
        logger.error("Creation of the directory %s failed : %s", outputFolder, error.strerror)
        sys.exit()
    #Change the current working directory:
    os.chdir(outputFolder)

    ######################################################
    # B) Parsing exonic intervals bed
    exonInterval=processBed(bedFilePath)
    exonInterval.to_csv(("ExonsIntervals_processBed_"+now),sep="\t", index=False, header=None)

    ######################################################
    # C) Creating interval trees for each chromosome
    dictIntervalTree=IntervalTreeDictCreation(exonInterval)

    ############################################
    # D] Parsing old counts file (.tsv)


    #####################################################
    # B)Bam files list extraction
    #This list contains the path to each bam.
    if not os.path.isdir(bamFilePath):
        logger.error("Bam folder doesn't exist %s",bamFilePath)
        sys.exit()
    
    samples=fnmatch.filter(os.listdir(bamFilePath), '*.bam')

    #####################################################
    # C) Canonical transcript bed, existence check    
    # Bed file load and Sanity Check : If the file does not have the expected format, the process ends.
    exonInterval=processBed(intervalFilePath)
    
    ############################################
    # D]  Creating interval trees for each chromosome
    dictIntervalTree=IntervalTreeDictCreation(exonInterval)

    ############################################
    # E] Old count file analysis if available


    #####################################################
    # F) Definition of a loop for each sample allowing the BAM files symlinks and the reads counting.
    for S in samples:
        sampleName=S.replace(".bam","")
        bamFile=os.path.join(bamFilePath,S)
        logger.info('Sample being processed : %s', sampleName)

        #output TSV name definition
        nameCountFilePath=os.path.join(RCPathOutput,sampleName+".tsv")
        validSanity=SanityCheckPreExistingTSV(nameCountFilePath, intervalFilePath,intervalBed)
        print(sampleName,"pre-existing correct count file :",validSanity)

        if validSanity:
            continue
        else:
            SampleCountingFrag(bamFile,nameCountFilePath,dictIntervalTree,intervalBed,processTmpDirPath,num_cores)
                

if __name__ =='__main__':
    main(sys.argv[1:])