#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############# STEP2 makeCNVCalls DECoN
#############################################################################################################
# How the script works ?
# This python script allows to launch the DECON tool (in R) modified to use the Bedtools results (script AdaptDECON_CallingCNV_BedtoolsRes.R)
# It will however allow to build the tree of the output files.
# It will also allow to realize the security checks on the input data.
#
# The calling is based on the selection of a set of the most correlated patients.
# This base profile is then compared to the target patient.
# A beta-binomial distribution is fitted to the data to extract the differences in copy number via a likelihood profile.
# The segmentation is performed using a hidden Markov chain model.
# The output results have a format identical to the basic DECON format. 


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
import FileFolderManagement.functions as fc

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
logger.propagate = False
##############################################################################################################
############### Script Body
##############################################################################################################

def main(argv):
	##########################################
	# A) ARGV parameters definition
    bedfile = ''
    readCountFolder=''
    outputfile = ''

    try:
	    opts, args = getopt.getopt(argv,"h:b:r:o:",["help","bedfile=","readCountFolder=","outputfile="])
    except getopt.GetoptError:
	    print('python3.6 STEP_2_GermlineCNVCaller_DECoN.py -bed <bedfile> -r <readCountFolder> -o <outputfile>')
	    sys.exit(2)
    for opt, value in opts:
	    if opt == '-h':
		    print("COMMAND SUMMARY:"			
		    +"\n This script is used to ..."
		    +"\n"
		    +"\n USAGE:"
		    +"\n python3.6 STEP_2_GermlineCNVCaller_DECoN.py -bed <bedfile> -r <readCountFolder> -o <outputfile> "
		    +"\n"
		    +"\n OPTIONS:"
		    +"\n	-b : .bed file that was used in the previous read counting step."
		    +"\n	-o : path to the folder containing the read counts for each patient."
		    +"\n	-n : path to the output folder.")
		    sys.exit()
	    elif opt in ("-b", "--bedfile"):
		    bedfile = value
	    elif opt in ("-r", "--readCountFolder"):
		    readCountFolder = value
	    elif opt in ("-o", "--outputfile"):
		    outputfile = value

    logger.info('Bedfile file is %s', bedfile)
    logger.info('ReadCountFolder is %s ', readCountFolder)
    logger.info('Output file is %s ', outputfile)

    #####################################################
    # B) GATK analysis output file creation 
    OutputAnalysisPath=fc.CreateAnalysisFolder(outputfile, "Calling_results_Bedtools_"+now )

    #####################################################
    # C) Existence File Check
    fc.checkFolderExistance(readCountFolder)
    fc.checkFileExistance(bedfile)

    #####################################################
    # D) use of the R script for calling
    DECoNCommand="Rscript /home/septiera/InfertilityCohort_Analysis/Scripts/Rscript/AdaptDECON_CallingCNV_BedtoolsRes.R "+readCountFolder+" " +bedfile+" "+OutputAnalysisPath
    logger.info("DECoN modified COMMAND: %s", DECoNCommand)
    returned_value = os.system(DECoNCommand)
    if returned_value == 0:
        logger.info("Command Line DECoN modified  MakeCNVCalls works well")
    else:
        logger.error("Command Line DECoN modified MakeCNVCalls don't work, error status = %s", returned_value)
        sys.exit()
					
if __name__ =='__main__':
    main(sys.argv[1:])


