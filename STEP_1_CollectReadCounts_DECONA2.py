#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############# STEP1 Collect Read Count
#############################################################################################################
# How the script works ?
# This script performs READS counts (in genomics intervals => exons) using the bedtools software from exome sequencing data.

#Input file:
#-the path to a file containing the intervals (delimitation of the exons), these intervals are paddled +-10pb , it is composed of 4 columns (CHR,START,END,TranscriptID_ExonNumber)
#-the path to the bam files. (cf Nicolas Thierry-Mieg's https://github.com/ntm/grexome-TIMC-Primary pipeline to know how they were obtained)
#-the path to the output files.

#Output files:
#The process is done sample by sample.
#A parallelization method has been implemented in order to process several samples at the same time (5 threads for memory optimization)
#The output file for a sample is a tsv of 5 columns without header (CHR, START, END, TranscriptID_ExonNumber, ReadCount)
#The read count is done in this script with bedtools 2 and its multibamCov program.
#The command implies that the reads have a mapq >= 20.

#############################################################################################################
############# Loading of the modules required for processing.
#############################################################################################################

import sys, getopt # this module provides system information. (ex argv argument)
import pysam #Is an add-on module to samtools to read bam files
import glob #Is a module that allows you to browse all the files 
import pandas as pd #is a module that makes it easier to process data in tabular form by formatting them in dataframes as in R.
import numpy #is a module often associated with the pandas library it allows the processing of matrix or tabular data.
import os # this module provides a portable way to use operating system-dependent functionality. (opening and saving files)
import re, fnmatch #is a module that allows you to search for specific strings using regular expressions
import time # is a module for obtaining the date and time of the system.
import logging # is a practical logging module. (process monitoring and development support)
import multiprocessing #this module allows to parallelize the process.
from joblib import Parallel, delayed #this module allowing to define the parallelization characteristics works with the multiprocessing module.

#Unique Modules Localization necessary to avoid version errors between different processes.
sys.path.append("/home/septiera/Benchmark_CNVTools/Scripts/Modules/")
import FileFolderManagement.functions as ffm #Module to check the existence of files
import InputSanityCheck.functions as isc #Sanity Check of input files

#Scripts execution date definition(allows to annotate the output files to track them over time) 
now=time.strftime("%y%m%d")

#############################################################################################################
############# Logging Definition 
#############################################################################################################
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
##############################################################################################################
############### Script Body
##############################################################################################################

def main(argv):
    ##########################################
    # A) ARGV parameters definition
    intervalFile = ''
    bamOrigFolder=''
    outputFile =''

    try:
        opts, args = getopt.getopt(argv,"h:i:b:o:",["help","intervalFile=","bamfolder=","outputfile="])
    except getopt.GetoptError:
        print('python3.6 STEP_1_CollectReadCounts_Bedtools.py -i <intervalFile> -b <bamfolder> -o <outputfile>')
        sys.exit(2)
    for opt, value in opts:
        if opt == '-h':
            print("COMMAND SUMMARY:"			
            +"\n This script performs READ counts using the bedtools software from exome sequencing data."
            +"\n"
            +"\n USAGE:"
            +"\n python3.6 STEP_1_CollectReadCounts_Bedtools.py -i <intervalfile> -b <bamfolder> -o <outputfile>"
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
    DataToProcessPath=ffm.CreateAnalysisFolder(outputFile,"DataToProcess")    
        # 2) Creation of the ReadCount folderif not existing.
    RCPath=ffm.CreateAnalysisFolder(DataToProcessPath,"ReadCount")
        # 3) Creation of the Bedtools folder if not existing.
    RCPathOutput=ffm.CreateAnalysisFolder(RCPath,"DECONA2")

    #####################################################
    # B)Bam files list extraction from Nico's original folder.
    #This list contains the path to each bam.
    ffm.checkFolderExistance(bamOrigFolder)
    bamlist=list()
    sample=fnmatch.filter(os.listdir(bamOrigFolder), '*.bam')

    #####################################################
    # C) Canonical transcript bed, existence check 
    ffm.checkFileExistance(intervalFile)
    
    #####################################################
    # D) Bed file load and Sanity Check : If the file does not have the expected format, the process ends.
    intervalBed=isc.BedSanityCheck(intervalFile, "TRUE") #The table will not be used but the verification of its format is necessary.
    CHRlist=list((intervalBed.CHR).unique())
    inputs=numpy.arange(0,len(CHRlist))
    #####################################################
    # E) Definition of a loop for each sample allowing the BAM files symlinks and the reads counting.
    def CountingFunction(i,intervalBed,bamFile, FinalTab):
        CHROM=CHRlist[i]
        print(CHROM)
        CHRBed=intervalBed.loc[intervalBed.CHR==CHROM,]
        for indexInt, rowInt in CHRBed.iterrows():
            rList2=[]
            S=rowInt.START
            E=rowInt.END
            ENST=rowInt.TranscriptID_ExonNumber
            HashReadStart={}
            HashReadEnd={}
            HashReadTLEN={}
            bam=pysam.AlignmentFile(bamFile, "rb")
            for read in bam.fetch(CHROM, start=S,stop=E):
                if (read.mapq >= 20
                    and not read.is_duplicate and not read.is_qcfail
                    and not read.is_secondary and not read.is_unmapped):
                    HashReadStart.setdefault(read.qname,[]).append(read.pos)
                    HashReadEnd.setdefault(read.qname,[]).append(read.reference_end)
                    HashReadTLEN.setdefault(read.qname,[]).append(read.tlen)
                    #rList2.append(read)

            ########
            ## Process
            QnameList=list(HashReadStart.keys())
            CountFrag=0
            CountRead=0
            for qname in QnameList:
                    #print(qname)
                    #Key initilization
                    Startpos=HashReadStart[qname]
                    Endpos=HashReadEnd[qname]
                    tlenpos=HashReadTLEN[qname]
                    #reads recovery
                    NbReads=len(Startpos)
                    #Calculation of the maximum and minimum bounds to know if there is an overlap on an exon      
                    minStart=min(Startpos)
                    maxEnd=max(Endpos)

                    #If there is an overlap, there are several cases to consider
                    ########################################################
                    # Case 1: the classic case of two paired reads
                    ########################################################
                    if NbReads==2:
                        ###########
                        #A) If they overlap
                        if int(Startpos[1])<=int(Endpos[0]) and int(Startpos[0])<= int(Endpos[1]):
                            #browse the intervals of the bed overlapping the concerned reads
                            CountFrag=CountFrag+1
                        ###########
                        #B)If they do not overlap
                        else :
                            CountRead=CountRead+2

                    #########################################################       
                    # Case n°2 : the case where there are split reads (more complex : combinations to realize)
                    ########################################################
                    elif NbReads>2:
                        #A) Création d'une liste pour supprimer les reads déjà appariés des comparaisons futures
                        listNotKeep=[]
                        # B) Parcour du premier read de la liste jusqu'a l'avant dernier (premiuer terme de la comparaison)
                        for cp1 in range(NbReads-1):
                            #1) si il reste suffisamment de comparaison possibles
                            if (len(set(range(NbReads))-set(listNotKeep))>1): 
                                R1start=Startpos[cp1]
                                R1end=Endpos[cp1]
                                R1TLEN=tlenpos[cp1]
                                # C) Parcour du deuxieme read (read1+1) jusqu'au dernier:
                                for cp2 in range(cp1+1,NbReads):
                                    # 1) si les read n'ont pas déjà été éliminés car déjà associés précédemment
                                    if (cp2 not in listNotKeep) and (cp1 not in listNotKeep):
                                        R2start=Startpos[cp2]
                                        R2end=Endpos[cp2]
                                        R2TLEN=tlenpos[cp2]
                                        #2)Si on respect le sens des appariemments
                                        if ((R1TLEN>0) and (R2TLEN<0)) or ((R1TLEN<0) and (R2TLEN>0)) :
                                            #############
                                            # a) If the two current reads overlap
                                            if  int(R2start)<=int(R1end) and int(R1start)<=int(R2end):
                                                CountFrag=CountFrag+1
                                                listNotKeep.append(cp1)
                                                listNotKeep.append(cp2)
                                            #############
                                            # b) If they do not overlap
                                            else:
                                                CountRead=CountRead+2
                                                listNotKeep.append(cp1)
                                                listNotKeep.append(cp2)
                        #2) Si un plusieurs read n ont pas trouvés de paires         
                        if len(set(range(NbReads))-set(listNotKeep))>=1:
                            CountRead=CountRead+len(set(range(NbReads))-set(listNotKeep))

                    ############################################################
                    # Cas n°3 : If only one reads because its pair has been filtered before
                    ########################################################
                    elif NbReads==1:
                         CountRead=CountRead+1

            TotalCount=CountFrag+CountRead
            #liste=[[CHROM,S, E,ENST,len(rList2),CountFrag,CountRead,TotalCount]]
            liste=[[CHROM,S, E,ENST,TotalCount]]
            FinalTab=FinalTab.append(pd.DataFrame(liste), ignore_index=True)
        #ChromTab=ChromTab.append(FinalTab, ignore_index=True)
        return(FinalTab)

    for s in sample:
        name=s.replace(".bam","")
        print(name)
        bamFile=os.path.join(bamOrigFolder,s)
        bam=pysam.AlignmentFile(bamFile, "rb")
        #processed_list = Parallel(n_jobs=num_cores)(delayed(my_function(i) for i in inputs)
        FinalTab=pd.DataFrame()
        results = Parallel(n_jobs=12)(delayed(CountingFunction)(i,intervalBed,bamFile,FinalTab) for i in numpy.arange(0,len(CHRlist)))
        results=pd.concat(results)
        results.to_csv(os.path.join(RCPathOutput,name+".tsv"),sep="\t", index=False, header=None)

if __name__ =='__main__':
    main(sys.argv[1:])
    

