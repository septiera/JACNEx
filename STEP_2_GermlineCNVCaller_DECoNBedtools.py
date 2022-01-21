#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############################################## STEP2 makeCNVCalls DECoN #####################################
#############################################################################################################
# How the script works ?

# This python script allows to launch the DECON tool (in R) modified to use the 
# Bedtools results or DECONA2 results (script AdaptDECON_CallingCNV_BedtoolsRes.R)
# Its main role is to carry out the CNVs calls.
# It will also allow to realize the security checks on the input data.
# There is only 1 entry to indicate:
#   -the path to the file containing the count for each patients (on columns preceded by
#    the 4 columns (CHR,START,END,EXON-ID)).

# The calling is based on the selection of a set of the most correlated patients.
# This base profile is then compared to the target patient.
# A beta-binomial distribution is fitted to the data to extract the differences in 
# copy number via a likelihood profile.
# The segmentation is performed using a hidden Markov chain model.
# The output results have a format identical to the basic DECON/ExomDepth format. 
# It corresponds to a .tsv table containing all the CNVs called for the samples set considered.
# The output file has 15 columns:
    # sample : sample name (str)
    # correlation : average correlation between the target sample and the control sample set 
    #               (The controls are samples of the data set, not real controls). (float)
    # N.Comp : control samples number used (int)
    # start.p + end.p : index corresponding to the line of exons affected by the CNV in the 
    #                   bed file (int)
    # type : cnv type ("duplication" or "deletion" str)
    # nexons : number of exons affected by the cnv (int)
    # start + end :  CNV genomic locations (WARN: deduced from the affected exons intervals 
    #               so no precise breakpoints)
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
"Give a count file (reads or fragments) on the canonical transcripts exonic intervals,\n"+
"for a minimum of 20 samples, the CNVs can be called.\n"+
"Call algorithm : beta-binomial. \n"+
"Segmentation : hidden Markov chain model. \n\n"+
"OPTIONS:\n"+
"    -c or --counts [str] optional: a .tsv count file (with path), contains 4 columns (CHR,START,END,EXON-ID)\n"+
"                                   corresponding to the target interval informations.\n"+
"                                   Each subsequent column is the counting for one sample of the set.\n\n"

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    ##########################################
    # A) Getopt user argument (ARGV) recovery
    countFile=""

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h', ["help","counts="])
    except getopt.GetoptError as e:
        print("ERROR : "+e.msg+".\n",file=sys.stderr)  
        usage()
        sys.exit(1)

    for opt, value in opts:
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ("-c","--counts"):
            countFile=value
        else:
            print("ERROR : Programming error. Unhandled option "+opt+".\n",file=sys.stderr)
            sys.exit(1)

    #####################################################
    # B) Checking that the parameter actually exist 
    if (countFile!="") and (not os.path.isfile(countFile)):
        sys.stderr.write("ERROR : Countfile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)

    #####################################################
    # C) CountFile Sanity Check 
    if (countFile!="") and (not os.path.isfile(countFile)):
        sys.stderr.write("ERROR : Countfile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)


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


