#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############# STEP0 IntervalList Initialisation
#############################################################################################################
# How the script works ?
#This script is used to generate exon interval files (.bed) for use with the DECON/ExomeDepth tool.
#Its use is described in the README of the scripts.

#The input entries are:
#-the path to a file containing the canonical transcripts (cf /home/nthierry/Transcripts_Data/README)
#-genome version string (ex: "GRCH38_v101")
#-the path to the output file

#This script may not produce a new bed if the previous version is identical to the one being processed.
#If a bed file must be generated:
#-a bed file containing all the chromosomes, the intervals for each exon have been padded +-10bp. This file contains 4 columns: CHR, START, END, ENST_EXON.
#The format of the "CHR" column is "chr/d".
#The ENST_EXON column comes from the canonical transcript file. No changes are made to it.
#Identical intervals covering exons of different genes are not affected.
#This file is tab delimited and has no column name.

#WARNING !!!: This script is to be done only when changing the reference genome (new version). 

#############################################################################################################
############# Loading of the modules required for processing.
#############################################################################################################
import pandas as pd #is a module that makes it easier to process data in tabular form by formatting them in dataframes as in R.
import numpy #is a module often associated with the pandas library it allows the processing of matrix or tabular data.
import os #this module provides a portable way to use operating system-dependent functionality. (opening and saving files)
import sys #this module provides system information. (ex argv argument)
import time #is a module for obtaining the date and time of the system.
import logging #is a practical logging module. (process monitoring and development support)
import getopt #this module provides system information. (ex argv argument)
import re , fnmatch #this module allows you to search and replace characters with regular expressions.

# Unique Modules Localization necessary to avoid version errors between different processes.
sys.path.append("/home/septiera/Benchmark_CNVTools/Scripts/Modules/") 
import FileFolderManagement.functions as ffm #Module to check the existence of files
import InputSanityCheck.functions as isc #Sanity Check of input files

#Definition of the scripts'execution date (allows to annotate the output files to track them over time) 
now=time.strftime("%y%m%d")

#############################################################################################################
############# Logging Definition 
#############################################################################################################
# create logger
logger = logging.getLogger(sys.argv[0])
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)-8s [%(process)d] %(message)s', '%Y-%m-%d %H:%M:%S')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

##############################################################################################################
############### Script Body
##############################################################################################################

def main(argv):
    ##########################################
    # A) ARGV parameters definition
    BedCanonicalTranscript =''
    GenomeVersion=''
    OutputFolder =''

    try:
        opts, args = getopt.getopt(argv,"h:b:n:o:",["help","BedFile=","GenomeName=","outputPath="])
    except getopt.GetoptError:
        print('python3.6 STEP_0_IntervalList.py -b <BedFile> -n <GenomeName> -o <outputPath>')
        sys.exit(2)
    for opt, value in opts:
        if opt == '-h':
            print("COMMAND SUMMARY:"			
            +"\n This script is used to generate exon interval files (.bed) for use with the DECON/ExomeDepth tool."
            +"\n"
            +"\n USAGE:"
            +"\n python3.6 STEP_0_IntervalList.py -b <BedFile> -n <GenomeName> -o <outputPath>"
            +"\n"
            +"\n OPTIONS:"
            +"\n	-b : The path to the canonicalTranscript.bed file."
            +"\n	-n : A string is expected to describe the reference genome version.Please write as follows for interoperability : 'GRCH38_v101'"
            +"\n	-o : The path where create the new output file for the current analysis.")
            sys.exit()
        elif opt in ("-b", "--BedFile"):
            BedCanonicalTranscript=value
        elif opt in ("-n", "--GenomeName"):
            GenomeVersion=value
        elif opt in ("-o", "--outputfile"):
            OutputFolder=value

    #Check that all the arguments are present.
    logger.info('Canonical Transcript bed path is %s', BedCanonicalTranscript)
    logger.info('Genome Names version is %s',  GenomeVersion)
    logger.info('Output file is %s ', OutputFolder)

    ###############################
    # A) Bed file to be completed,existence check 
    ffm.checkFileExistance(BedCanonicalTranscript)
    #In general, if the file does not exist, this corresponds to a change in the genome version.

    #####################
    # B) Bed file load and Sanity Check
    BedToComplete=isc.BedSanityCheck(BedCanonicalTranscript,"TRUE")
    
    ##############################
    # C) Sorting the lines according to their genomic position.
    #X and Y annotations replacement for the sex chromosomes by their numerical values (23 and 24)
    #chrM = 25
    BedToComplete["CHR1"]=BedToComplete["CHR"]
    BedToComplete.CHR1=BedToComplete.CHR1.astype(str)
    BedToComplete["CHR1"]=[re.sub("chr", "",x) for x in BedToComplete["CHR1"]]
    BedToComplete.loc[BedToComplete["CHR1"]=='X',"CHR1"]=23
    BedToComplete.loc[BedToComplete["CHR1"]=='Y',"CHR1"]=24
    BedToComplete.loc[BedToComplete["CHR1"]=='M',"CHR1"]=25
    BedToComplete.CHR1=BedToComplete.CHR1.astype(int)

    #Primary sorting = CHR, secondary sorting = START, thirdly sorting= END
    BedToComplete=BedToComplete.sort_values(by=["CHR1","START","END"])
    BedToComplete=BedToComplete[["CHR","START","END","TranscriptID_ExonNumber"]]
    logger.info("The table sorting stage works well.")

    ######################################
    # D) Padding File
    BED=BedToComplete
    BED["START"]=BED["START"]-10
    BED["END"]=BED["END"]+10

    ######################################
    # E) Writing annotation files.
    #StaticFolder existence check:
    PathStaticFolder=ffm.CreateAnalysisFolder(OutputFolder,"StaticFiles")

    #The different conditions below allows to search old interval files and compare them to the new one.
    #However, the folder must contain only one old file. 
    #If this is not the case, the user must select only one version.
    #If the old file is exactly the same as the new one, no new version is generated.
    #If it's not identical the old version is moved to the OLD folder and a new one is created.
    NameBed="STEP0_*"
    listFilesBed=fnmatch.filter(os.listdir(os.path.join(PathStaticFolder)), NameBed)
    if len(listFilesBed)==1:
        logger.info("Presence of an old interval file %s.",listFilesBed[0])
        OldTable=isc.BedSanityCheck(os.path.join(PathStaticFolder,listFilesBed[0]),"TRUE")
        if (len(BED)!=len(OldTable)):
            logger.warning("The two files are not identical in terms of lines number.")
            CommandBash="mv "+os.path.join(PathStaticFolder,listFilesBed[0])+" "+os.path.join(PathStaticFolder,OLD)
            returned_value=os.system(CommandLine)
            #Security Check
            if returned_value == 0:
                logger.info("File %s has been successfully moved.",listFilesBed[0])
            else:
                logger.error("File move has encountered an error %s.", CommandLine)
                sys.exit()
        else:
            startComp=BED.START[BED.START.isin(OldTable.START)]
            endComp=BED.END[BED.END.isin(OldTable.END)]
            if  len(startComp)!=len(OldTable) & len(endComp)!=len(OldTable) :
                logger.warning("The two interval files are not identical padding.")
                CommandBash="mv "+os.path.join(PathStaticFolder,listFilesBed[0])+" "+os.path.join(PathStaticFolder,OLD)
                returned_value=os.system(CommandLine)
                #Security Check
                if returned_value == 0:
                    logger.info("File %s has been successfully moved.",listFilesBed[0])
                else:
                    logger.error("File move has encountered an error %s.", CommandLine)
                    sys.exit()
            else:
                logger.info("It is not necessary to overwrite the old version of bed since it is identical to the new one. Therefore no creation of a new version.")
                sys.exit()
    elif len(listFilesBed)>1:
        logger.error("Several versions of the bed already exist. Please check them and keep only one if possible.")
        sys.exit()
    else:
        logger.info("There is no current version of the bed. This one can be created.")
        BED.to_csv(os.path.join(PathStaticFolder,"STEP0_"+GenomeVersion+"_Padding10pb_"+str(len(BedToComplete))+"exons_"+str(now)+".bed"),index=False, header=False,sep="\t")

if __name__ =='__main__':
    main(sys.argv[1:])
    
