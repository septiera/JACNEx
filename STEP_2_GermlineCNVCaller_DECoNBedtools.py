#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############################################## STEP2 makeCNVCalls DECoN #####################################
#############################################################################################################
#STEPS:
#   A) Getopt user argument (ARGV) recovery
#   B) Checking that the mandatory parameters are presents
#   C) Check that the optional arguments are entered together.
#   D) Checking that the parameters actually exists
#   E) Using the R script based on the CNV calling part of ExomeDepth v1.1.15.

# The calling is based on the selection of a set of the most correlated patients.
# This base profile is then compared to the target patient.
# A beta-binomial distribution is fitted to the data to extract the differences in 
# copy number via a likelihood profile.
# The segmentation is performed using a hidden Markov chain model.
# The output results have a format identical to the basic ExomDepth format. 
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
import os
import getopt 
import logging 

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
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                                '%Y-%m-%d %H:%M:%S')
#add formatter to stderr handler
stderr.setFormatter(formatter)
#add stderr handler to logger
logger.addHandler(stderr)

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    ##########################################
    # A) Getopt user argument (ARGV) recovery
    countFile=""
    NEWcallFile=""
    NEWrefFile=""
    # default setting ARGV 
    OLDcallFile=""
    OLDrefFile=""

    usage="""\nCOMMAND SUMMARY:
Give a count file (reads or fragments) on the canonical transcripts exonic intervals,
for a minimum of 20 samples, the CNVs can be called.
Call algorithm : beta-binomial. 
Segmentation : hidden Markov chain model.
The script will print on stdout folder :
   -a new call file in TSV format, modifies the data from the pre-existing file
    if it is provided.
   -a new text tab-delimited file, modifies the data from the pre-existing file
    if it is provided.
OPTIONS:
    --counts [str] : a .tsv count file (with path), contains 4 columns (CHR,START,END,EXON-ID)
                     corresponding to the target interval informations.
    --outputcalls [str] : a new CNV calling file (with path).
    --outputref [str] : a new tab-delimited text file (with path)
                             Each subsequent column is the counting for one sample of the set 
                             (colnames=sampleNames).
    --calls [str] optional : a pre-parsed call file (with path), old cnv call lines are completed or
                            re-parsed if one or more new patients are added.
    --refs [str] optional :  a tab-delimited text file (with path)
                                 column 1: name of the sample to be processed,
                                 column 2: comma-delimited ref sample names list.\n"""

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","outputcalls=","outputrefs=","calls=","refs="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--counts"):
            countFile=value
        elif opt in ("--outputcalls"):
            NEWcallFile=value
        elif opt in ("--outputrefs"):
            NEWrefFile=value
        elif opt in ("--calls"):
            OLDcallFile=value
        elif opt in ("--refs"):
            OLDrefFile=value
        else:
            sys.exit("ERROR : Programming error. Unhandled option "+opt+".\n")

    #####################################################
    # B) Checking that the mandatory parameter is present
    if countFile=="":
        print("ERROR :You must use --counts.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    if (NEWcallFile==""):
        print("ERROR :You must use --outputcalls.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    if (NEWrefFile==""):
        print("ERROR :You must use --outputref.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    #####################################################
    # C) Check that the optional arguments are entered together.
    if (OLDcallFile!="" and OLDrefFile=="") or (OLDcallFile=="" and OLDrefFile!=""):
        print("ERROR :If one of the optional arguments is entered the second is required.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    #####################################################
    # D) Checking that the parameters actually exists
    if not os.path.isfile(countFile):
        sys.exit("ERROR : Countfile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n") 
    if os.path.isfile(NEWcallFile):
       sys.exit("ERROR : new Callfile "+NEWcallFile+" already exist. Try "+scriptName+" --help.\n") 
    if os.path.isfile(NEWrefFile):
        sys.exit("ERROR : new Callfile "+NEWrefFile+" already exist. Try "+scriptName+" --help.\n") 

    if (OLDcallFile!="") and (not os.path.isfile(OLDcallFile)):
        sys.exit("ERROR : CallFile "+OLDcallFile+" doesn't exist. Try "+scriptName+" --help.\n")
    if (OLDrefFile!="") and (not os.path.isfile(OLDrefFile)):
        sys.exit("ERROR : CallFile "+OLDrefFile+" doesn't exist. Try "+scriptName+" --help.\n") 

    #####################################################
    # E) Use of the R script for calling
    if OLDrefFile=="":
        EDCommand="Rscript ./Rscript/CallCNV_ExomeDepth.R "+countFile+" "+NEWcallFile+" "+NEWrefFile+""
        logger.info("\nExomeDepth  COMMAND: %s \n", EDCommand)
        returned_value = os.system(EDCommand)
        if returned_value == 0:
            logger.info("Command Line ExomeDepth CNVCalls works well\n")
        else:
            logger.error("Command Line ExomeDepth  CNVCalls don't work, error status = %s \n", returned_value)
            sys.exit()
    else:
        EDCommand="Rscript ./Rscript/CallCNV_ExomeDepth.R "+countFile+" "+NEWcallFile+" "+NEWrefFile+" "+OLDcallFile+" "+OLDrefFile+""
        logger.info("\nExomeDepth COMMAND: %s\n", EDCommand)
        returned_value = os.system(EDCommand)
        if returned_value == 0:
            logger.info("Command Line ExomeDepth CNVCalls works well \n")
        else:
            logger.error("Command Line ExomeDepth  CNVCalls don't work, error status = %s \n", returned_value)
            sys.exit()			

if __name__ =='__main__':
    main()