#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############# STEP2 makeCNVCalls DECoN
#############################################################################################################
# How the script works ?

# This python script allows to launch the DECON tool (in R) modified to use the Bedtools results (script AdaptDECON_CallingCNV_BedtoolsRes.R)
# Its main role is to carry out the CNVs calls.
# It will however allow to build the tree of the output files.
# It will also allow to realize the security checks on the input data.
# The files needed for the analysis are loaded independently.(.bed, output)
# There are 3 entries to indicate:
#   -the path to the folder containing the read count files.
#   -the path to a file containing the intervals (delimitation of the exons), these intervals are paddled +-10pb , it is composed of 4 columns (CHR,START,END,TranscriptID_ExonNumber)
#   -the path to the outputs.

# The calling is based on the selection of a set of the most correlated patients.
# This base profile is then compared to the target patient.
# A beta-binomial distribution is fitted to the data to extract the differences in copy number via a likelihood profile.
# The segmentation is performed using a hidden Markov chain model.
# The output results have a format identical to the basic DECON format. 
# It corresponds to a .tsv table containing all the CNVs called for the samples set considered.

#############################################################################################################
############# Loading of the modules required for processing.
#############################################################################################################
import pandas as pd #is a module that makes it easier to process data in tabular form by formatting them in dataframes as in R.
import numpy #is a module often associated with the pandas library it allows the processing of matrix or tabular data.
import os # this module provides a portable way to use operating system-dependent functionality. (opening and saving files)
import sys, getopt # this module provides system information. (ex argv argument)
import time # is a module for obtaining the date and time of the system.
import logging # is a practical logging module. (process monitoring and development support)
sys.path.append("/home/septiera/Benchmark_CNVTools/Scripts/Modules/")
import FileFolderManagement.functions as ffm #Module to check the existence of files

#Definition of the scripts'execution date (allows to annotate the output files to track them over time) 
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
    readCountFolder=''
    outputfile = ''

    try:
	    opts, args = getopt.getopt(argv,"h:i:r:o:",["help","intervalFile=","readCountFolder=","outputfile="])
    except getopt.GetoptError:
	    print('python3.6 STEP_2_GermlineCNVCaller_DECoNBedtools.py -i <intervalFile> -r <readCountFolder> -o <outputfile>')
	    sys.exit(2)
    for opt, value in opts:
	    if opt == '-h':
		    print("COMMAND SUMMARY:"			
		    +"\n This python script allows to launch the DECON tool (in R) modified to use the Bedtools results (script AdaptDECON_CallingCNV_BedtoolsRes.R)"
            +"\n Its main role is to carry out the CNVs calls."
		    +"\n"
		    +"\n USAGE:"
		    +"\n python3.6 STEP_2_GermlineCNVCaller_DECoNDECoNBedtools.py -i <intervalFile> -r <readCountFolder> -o <outputfile> "
		    +"\n"
		    +"\n OPTIONS:"
		    +"\n	-i : A bed file obtained in STEP0. Please indicate the full path.(4 columns : CHR, START, END, TranscriptID_ExonNumber)"
		    +"\n	-o : path to the folder containing the read counts for each patient."
		    +"\n	-n : path to the output folder.")
		    sys.exit()
	    elif opt in ("-i", "--intervalFile"):
		    intervalFile = value
	    elif opt in ("-r", "--readCountFolder"):
		    readCountFolder = value
	    elif opt in ("-o", "--outputfile"):
		    outputfile = value

    #Check that all the arguments are present.
    logger.info('Intervals bed file path is %s', intervalFile)
    logger.info('ReadCount folder path is %s ', readCountFolder)
    logger.info('Output file path is %s ', outputfile)

    #####################################################
    # B) Bedtools analysis output file creation 
    OutputAnalysisPath=ffm.CreateAnalysisFolder(outputfile, "Calling_results_Bedtools_"+now )

    #####################################################
    # C) Existence File Check
    ffm.checkFolderExistance(readCountFolder)
    ffm.checkFileExistance(intervalFile)

    #####################################################
    # D) Use of the R script for calling
    DECoNCommand="Rscript /home/septiera/InfertilityCohort_Analysis/Scripts/Rscript/AdaptDECON_CallingCNV_BedtoolsRes.R "+readCountFolder+" "+intervalFile+" "+OutputAnalysisPath
    logger.info("DECoN modified COMMAND: %s", DECoNCommand)
    returned_value = os.system(DECoNCommand)
    if returned_value == 0:
        logger.info("Command Line DECoN modified  MakeCNVCalls works well")
    else:
        logger.error("Command Line DECoN modified MakeCNVCalls don't work, error status = %s", returned_value)
        sys.exit()
					
if __name__ =='__main__':
    main(sys.argv[1:])


