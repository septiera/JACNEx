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
# The output file has 16 columns:
    # sample : sample name (str)
    # correlation : average correlation between the target sample and the control sample set 
    #               (The controls are samples of the data set, not real controls). (float)
    # N.Comp : control samples number used (int)
    # SampN.Comp : control samples names used (str)
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
"The script will print on stdout folder :\n"+
"   -a new call file in TSV format, modifies the data from the pre-existing file\n"+
"    if it is provided.\n"+
"   -a new text tab-delimited file, modifies the data from the pre-existing file\n"+
"    if it is provided.\n"+
"OPTIONS:\n"+
"    --counts [str] : a .tsv count file (with path), contains 4 columns (CHR,START,END,EXON-ID)\n"+
"                     corresponding to the target interval informations.\n"+
"    --outputcalls [str] : a new CNV calling file (with path).\n"
"    --outputcontrols [str] : a new tab-delimited text file (with path)\n"+
"                     Each subsequent column is the counting for one sample of the set (colnames=sampleNames).\n\n"
"    --calls [str] optional : a pre-parsed call file (with path), old cnv call lines are completed or\n"+
"                            re-parsed if one or more new patients are added.\n"
"    --controls [str] optional :  a tab-delimited text file (with path)\n"+
"                                column 1: name of the sample to be processed,\n"+
"                                column 2: comma-delimited list of control sample names.\n\n")

#################################################
# parseCountFile :
#Input:
# - countFile is a tsv file (with path), including column titles, as
#   specified previously
#
# -> Parse countFile into a dataframe (will be returned)
# -> Check that the first 4 columns are identical to exons,
#    otherwise die with an error.
# -> check that the samples counts columns are [int] type.

def parseCountFile(countFile):
    try:
        counts=pd.read_table(countFile,sep="\t")
    except Exception as e:
        logger.error("Parsing provided countFile %s: %s", countFile, e)
        sys.exit(1)

    if not (counts.dtypes["CHR"]=="O" and counts.dtypes["EXON_ID"]=="O"):
        logger.error("One or both of the 'CHR' and 'EXON_ID' columns are not in the correct format. Please check it.\n"
                    +"The column must be a python object [str]")
        sys.exit(1)
    elif not (counts.dtypes["START"]=="int64" and counts.dtypes["END"]=="int64"):
        logger.error("One or both of the 'START' and 'END' columns are not in the correct format. Please check it.\n"
                    +"The columns must contain integers.")
    else: 
        #check that the old sample data is in [int] format.
        namesSampleToCheck=[]
        for columnIndex in range(5,len(counts.columns)):
            if not counts.iloc[:,columnIndex].dtypes=="int64":
                namesSampleToCheck.append(counts.iloc[:,columnIndex].columns)
        if len(namesSampleToCheck)>0:
            logger.error("Columns in %s, sample(s) %s are not in [int] format.\n"+
            "Please check and correct these before trying again.", countFile,(",".join(namesSampleToCheck)))
            sys.exit(1)

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    ##########################################
    # A) Getopt user argument (ARGV) recovery
    countFile=""
    NEWcallFile=""
    NEWcontrolFile=""
    # default setting ARGV 
    OLDcallFile=""
    OLDcontrolFile=""

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h', ["help","counts=","calls=","controls=","outputcalls=","outputcontrols="])
    except getopt.GetoptError as e:
        print("ERROR : "+e.msg+".\n",file=sys.stderr)  
        usage()
        sys.exit(1)

    for opt, value in opts:
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ("--counts"):
            countFile=value
        elif opt in ("--outputcalls"):
            NEWcallFile=value
        elif opt in ("--outputcontrols"):
            NEWcontrolFile=value
        elif opt in ("--calls"):
            OLDcallFile=value
        elif opt in ("--controls"):
            OLDcontrolFile=value
        else:
            print("ERROR : Programming error. Unhandled option "+opt+".\n",file=sys.stderr)
            sys.exit(1)

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

    if (NEWcontrolFile==""):
        print("ERROR :You must use --outputcontrols.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    #####################################################
    # C) Check that the optional arguments are entered together.
    if (OLDcallFile!="" and OLDcontrolFile=="") or (OLDcallFile=="" and OLDcontrolFile!=""):
        print("ERROR :If one of the optional arguments is entered the second is required.\n",file=sys.stderr)
        usage()
        sys.exit(1)

    #####################################################
    # D) Checking that the parameters actually exists
    if not os.path.isfile(countFile):
        print("ERROR : Countfile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)
    if os.path.isfile(NEWcallFile):
        print("ERROR : new Callfile "+NEWcallFile+" already exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1) 
    if os.path.isfile(NEWcontrolFile):
        print("ERROR : new Callfile "+NEWcontrolFile+" already exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)

    if (OLDcallFile!="") and (not os.path.isfile(OLDcallFile)):
        print("ERROR : CallFile "+OLDcallFile+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)
    if (OLDcontrolFile!="") and (not os.path.isfile(OLDcontrolFile)):
        print("ERROR : CallFile "+OLDcontrolFile+" doesn't exist. Try "+scriptName+" --help.\n",file=sys.stderr) 
        sys.exit(1)

    #####################################################
    # C) CountFile Sanity Check 
    parseCountFile(countFile)

    #####################################################
    # D) Use of the R script for calling
    if OLDcontrolFile=="":
        DECoNCommand="Rscript /Rscript/AdaptDECON_CallingCNV_BedtoolsRes.R "+countFile+" "+NEWcallFile+" "+NEWcontrolFile+""
        logger.info("DECoN modified COMMAND: %s", DECoNCommand)
        returned_value = os.system(DECoNCommand)
        if returned_value == 0:
            logger.info("Command Line DECoN modified MakeCNVCalls works well")
        else:
            logger.error("Command Line DECoN modified MakeCNVCalls don't work, error status = %s", returned_value)
            sys.exit()
    else:
        DECoNCommand="Rscript /Rscript/AdaptDECON_CallingCNV_BedtoolsRes.R "+countFile+" "+NEWcallFile+" "+NEWcontrolFile+" "+OLDcallFile+" "+OLDcontrolFile+""
        logger.info("DECoN modified COMMAND: %s", DECoNCommand)
        returned_value = os.system(DECoNCommand)
        if returned_value == 0:
            logger.info("Command Line DECoN modified MakeCNVCalls works well")
        else:
            logger.error("Command Line DECoN modified MakeCNVCalls don't work, error status = %s", returned_value)
            sys.exit()			

if __name__ =='__main__':
    main()


