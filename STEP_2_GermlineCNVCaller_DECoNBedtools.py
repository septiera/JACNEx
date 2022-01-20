#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############################################## STEP2 makeCNVCalls DECoN #####################################
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
# The output results have a format identical to the basic DECON/ExomDepth format. 
# It corresponds to a .tsv table containing all the CNVs called for the samples set considered.
# The output file has 15 columns:
    # sample : sample name (str)
    # correlation : average correlation between the target sample and the control sample set (The controls are samples of the data set, not real controls). (float)
    # N.Comp : control samples number used (int)
    # start.p + end.p : index corresponding to the line of exons affected by the CNV in the bed file (int)
    # type : cnv type ("duplication" or "deletion" str)
    # nexons : number of exons affected by the cnv (int)
    # start + end :  CNV genomic locations (WARN: deduced from the affected exons intervals so no precise breakpoints)
    # chromosome : without "chr" before (str)
    # id : "chr:start-end" , chr correspond to "chr"+str (str)
    # BF : Bayesian factor (float)
    # reads_expected : average of reads covering the interval for all control samples (int)
    # reads_observed : reads observed in the target patient for the interval 
    # reads_ratio : reads_observed/reads_expected (float)

#############################################################################################################
################################ Loading of the modules required for processing #############################
#############################################################################################################
import sys #path
import pandas as pd #read,make,treat Dataframe object
import os
import getopt 
import time # system date 
import logging 

#Definition of the scripts'execution date (allows to annotate the output files to track them over time) 
now=time.strftime("%y%m%d")

#####################################################################################################
################################ Logging Definition #################################################
#####################################################################################################
#create logger : Loggers expose the interface that the application code uses directly
logger=logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.DEBUG)
#create console handler and set level to debug : The handlers send the log entries (created by the loggers) to the desired destinations.
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
#create formatter : Formatters specify the structure of the log entry in the final output.
formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)-8s [%(process)d] %(message)s', '%Y-%m-%d %H:%M:%S')
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
        
##############################################################################################################
######################################### Script Body ########################################################
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
            +"\n	-r : path to the folder containing the read counts for each patient."
            +"\n	-o : path to the output folder.")
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
    outputFolderPath=outputfile+"Calling_results_Bedtools_"+now 
    CreateFolder(outputFolderPath)

    #####################################################
    # C) Existence File/Folder Check
    if not os.path.isdir(readCountFolder):
        logger.error("Bam folder doesn't exist %s",readCountFolder)
        sys.exit()

    if not os.path.isfile(intervalFile):
        logger.error("file %s doesn't exist ",intervalFile)
        sys.exit()

    #####################################################
    # D) Use of the R script for calling
    DECoNCommand="Rscript /home/septiera/InfertilityCohort_Analysis/Scripts/Rscript/AdaptDECON_CallingCNV_BedtoolsRes.R "+readCountFolder+" "+intervalFile+" "+outputFolderPath
    logger.info("DECoN modified COMMAND: %s", DECoNCommand)
    returned_value = os.system(DECoNCommand)
    if returned_value == 0:
        logger.info("Command Line DECoN modified  MakeCNVCalls works well")
    else:
        logger.error("Command Line DECoN modified MakeCNVCalls don't work, error status = %s", returned_value)
        sys.exit()
					
if __name__ =='__main__':
    main(sys.argv[1:])


