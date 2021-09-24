#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############# STEP0 IntervalList Initialisation
#############################################################################################################
# How the script works ?
#This script is used to generate exon interval files (.bed) for use with the DECON/ExomeDepth tool.
#Its use is described in the README of the scripts.

#############################################################################################################
############# Loading of the modules required for processing.
#############################################################################################################

import pandas as pd #is a module that makes it easier to process data in tabular form by formatting them in dataframes as in R.
import numpy #is a module often associated with the pandas library it allows the processing of matrix or tabular data.
import os # this module provides a portable way to use operating system-dependent functionality. (opening and saving files)
import sys # this module provides system information. (ex argv argument)
import time # is a module for obtaining the date and time of the system.
import logging # is a practical logging module. (process monitoring and development support)
sys.path.append("/home/septiera/Benchmark_CNVTools/Scripts/Modules/")
import FileFolderManagement.functions as ffm 

#Definition of the scripts'execution date (allows to annotate the output files to track them over time) 
now=time.strftime("%y%m%d")

#############################################################################################################
############# Logging Definition 
#############################################################################################################
logging.basicConfig(level=logging.DEBUG)
# create logger
logger=logging.getLogger(sys.argv[0])
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s  %(name)s : %(levelname)s - %(message)s', "%y%m%d %H:%M:%S")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
logging.getLogger().addHandler(ch)
logger.propagate=False
#############################################################################################################
############# Initiales files  
#############################################################################################################
#Please note that the access paths must be checked.

#Bed extracted from Nicolas' data (cf : /home/nthierry/Transcripts_Data/README) (4 columns : CHR Start End ENSEMBID_NBEX)
#Particularity of this bed: it contains all the intervals for each exons of the canonical transcripts of the reference genome Grch38_p101.  (294 153 exons)
PathBedToComplete="/data/septiera/REF_GENOME/canonicalTranscripts_210826.bed.gz"

#Bed from the gtf Grch38_p101 (cf : /data/septiera/REF_GENOME/README) (12 columns :"CHR1","START2","END3","DIR","ENSEMBLID","EXON","Gene_Name","gene_biotype","source","exon_version","tag","transcript_support_level")
#Contains all the annotations of the canonical transcripts and more.
PathBedWithAnnotation="/data/septiera/REF_GENOME/GRCh38.104_exonOnly_ForAnnotationSegments_210922.bed"

#Path to the static results files folder (normally the generated files do not need to be changed often)
OutputFolder="/home/septiera/InfertilityCohort_Analysis/Scripts/"

#Indication of a dictionary file to be used necessary for the picard BedToInterval command to obtain an interval_list file.
Dict="/data/septiera/REF_GENOME/hs38DH.dict"

# Fasta file path definition for the reference genome useful to apply the preprocessedInterval GATK4 command.
Fasta="/data/septiera/REF_GENOME/hs38DH.fa"

##############################################################################################################
############### Script Body
##############################################################################################################

###############################
# A) Bed file to be completed,existence check 
ffm.checkFileExistance(PathBedToComplete)
#####################
#Bed file to be completed, reading and formatting.
#### Table reading and transformation into dataframe.
BedToComplete=pd.read_table(PathBedToComplete,header=None)
#### Columns renaming (Warning : the columns number must be equal to 4).
if len(BedToComplete.columns) == 4:
    logger.info("BedToComplete contains 4 columns.")
    BedToComplete.columns=["CHR","START","END","IDTrans"]
else:
    logger.error("BedToComplete doesn't contains 4 columns:"
                 +"\n CHR, START, END, ENSEMBLID_EXON."
                 +"\n Please check the file before launching the script.")
    sys.exit()


##############################
# B) Bed file with annotation, existence check
ffm.checkFileExistance(PathBedWithAnnotation)
#####################
#Bed file with annotation, reading and formatting.
BedWithAnnotation=pd.read_table(PathBedWithAnnotation,header=None,dtype='unicode')
#### Columns renaming (Warning : the columns number must be equal to 12).
if len(BedWithAnnotation.columns) == 12:
    logger.info("BedWithAnnotation contains 12 columns")
    BedWithAnnotation.columns=["CHR1","START2","END3","DIR","ENSEMBLID","EXON","Gene_Name","gene_biotype","source","exon_version","tag","transcript_support_level"]
else:
    logger.error("BedWithAnnotation doesn't contains 12 columns:"
                 +"\n CHR, START, END, DIR, ENSEMBLID, EXON, Gene_Name, gene_biotype, source, exon_version, tag, transcript_support_level."
                 +"\n Please check the file before launching the script.")
    sys.exit()
    
##############################
# C) Creation of an extra column to combine ENSEMBLID transcript and EXON to help merge tables on a single key (like a dictionary, but since the bed files are read only once, no waste of time).
BedWithAnnotation["IDTrans"]=BedWithAnnotation.ENSEMBLID+"_"+BedWithAnnotation.EXON.astype(str)

#Two tables merge (Warning the merger is based on the number of lines of BedWithAnnotation also on the 294 153 exons).)
MergeTable=BedToComplete.merge(BedWithAnnotation,how='left',on="IDTrans")

#Check that the annotation has been carried out correctly by comparing the starts for each line.
test=MergeTable[~MergeTable.START.isin(MergeTable.START2)]
if len(test)==0:
    logger.info("The fusion between the BedToComplet and the BedWithAnnotation works well")
else:
    logger.error("The fusion between the BedToComplet and the BedWithAnnotation didn't works well."
                +"\n Difference between the starts of the intervals defining the exons."
                +"\n Please check the reference genome version for both bed files before restarting the script.")
    sys.exist()

##############################
# D) Sorting the lines according to their genomic position.
#X and Y annotations replacement for the sex chromosomes by their numerical values (23 and 24)
MergeTable.loc[MergeTable["CHR1"]=='X',"CHR1"]=23
MergeTable.loc[MergeTable["CHR1"]=='Y',"CHR1"]=24
MergeTable.loc[MergeTable["CHR1"]=='MT',"CHR1"]=25
MergeTable.CHR1=MergeTable.CHR1.astype(int)
MergeTable.START2=MergeTable.START2.astype(int)
MergeTable.END3=MergeTable.END3.astype(int)
MergeTable.START=MergeTable.START.astype(int)
MergeTable.END=MergeTable.END.astype(int)

#Primary sorting = CHR, secondary sorting = START.
MergeTable=MergeTable.sort_values(by=["CHR1","START2","END3"])
MergeTable=MergeTable[["CHR","START","END","DIR","ENSEMBLID","EXON","Gene_Name"]]
logger.info("The sorting stage of the table works well.")

######################################
# E) Padding File
BED=MergeTable
BED["START"]=BED["START"]-10
BED["END"]=BED["END"]+10

######################################
# F) Writing annotation files.
#Writing annotation files.
#StaticFolder existence check:
PathStaticFolder=ffm.CreateAnalysisFolder(OutputFolder,"StaticFiles")

#1) Annotation file containing all the intervals of the exons of canonical transcripts (even the overlapping ones).
MergeTable.to_csv(os.path.join(PathStaticFolder,"STEP0_Annotation_GRCH38_101_InitialData_"+str(len(MergeTable))+"exons_"+str(now)+".tsv"),index=False,sep="\t")

#2) Annotation file containing the padding intervals 10pb. 
BED.to_csv(os.path.join(PathStaticFolder,"STEP0_Annotation_GRCH38_101_Padding10pb_"+str(len(BED))+"exons_"+str(now)+".tsv"),index=False,sep="\t")

#3) Bed file containing the following indications:
#CHR:START:END
NameClassicBEDOv=os.path.join(PathStaticFolder,"STEP0_GRCH38_101_Padding10pb_"+str(len(BED))+"exons_"+str(now)+".bed")
ClassicBEDOv=BED[["CHR","START","END","ENSEMBLID"]]
ClassicBEDOv.to_csv(NameClassicBEDOv,index=False, header=False,sep="\t")

# 4) Deletion of duplicated exons because several genes have sequence homology, hence the duplication of exon intervals. (ex )
BED["compare"]=BED.CHR+"_"+BED.START.astype(str)+"_"+BED.END.astype(str)
BEDNoDup=BED.drop_duplicates(subset="compare")
BEDNoDup.to_csv(os.path.join(PathStaticFolder,"STEP0_Annotation_GRCH38_101_Padding10pb_NoDupExons_"+str(len(BEDNoDup))+"exons_"+str(now)+".tsv"),index=False,sep="\t")

#CHR:START:END
NameClassicBEDNoDup=os.path.join(PathStaticFolder,"STEP0_GRCH38_101_Padding10pb_NoDupExons_"+str(len(BEDNoDup))+"exons_"+str(now)+".bed")
ClassicBEDNoDup=BEDNoDup[["CHR","START","END","ENSEMBLID"]]
ClassicBEDNoDup.to_csv(NameClassicBEDNoDup,index=False, header=False,sep="\t")



