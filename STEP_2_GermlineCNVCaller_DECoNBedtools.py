#############################################################################################################
############################################## STEP2 makeCNVCalls DECoN #####################################
#############################################################################################################
# Performs CNV calling from exome fragment count data using ExomeDepth(v1.1.15).
# The calling is based on the most correlated patients set selection.
# Unlike the default ExomeDepth application, we splitted count data between gonosomes and autosomes to obtain
# optimal reference sets (e.g: Grouping same gender samples for gonosomes eliminates false positive calls).
# The base profile (sums the selected samples across each exon) is then compared to the target patient.
# A beta-binomial distribution is fitted to the data to extract the differences in 
# copy number via a likelihood profile.
# The segmentation is performed using a hidden Markov chain model.

#Inputs:
#   - a fragment count TSV file (possibly gzipped) : first 4 columns hold the exon definitions ("CHR","START","END","EXONID"),
#   subsequent columns (one per sample) hold the counts.
#   - a path defining the new calling TSV file (str) 
#   - a path defining the new reference TSV file (str) 
#   optional :
#   - a path to an old calling TSV file (str)
#   - a path to an old reference TSV file (str)

# Two output file are provided:
#   - The calls results have a format identical to the basic ExomDepth format. 
#   It corresponds to a TSV table containing all the CNVs called for the samples set considered.
#   The output file has 15 columns:
#       -sample : sample name (str)
#       -correlation : average correlation between the target sample and the control sample set 
#                   (The controls are samples of the data set, not real controls). (float)
#       -N.Comp : control samples number used (int)
#       -start.p + end.p : index corresponding to the line of exons affected by the CNV in the 
#                       bed file (int)
#       -type : cnv type ("duplication" or "deletion" str)
#       -nexons : number of exons affected by the cnv (int)
#       -start + end :  CNV genomic locations (WARN: deduced from the affected exons intervals 
#                   so no precise breakpoints)
#       -chromosome : without "chr" before (str)
#       -id : "chr:start-end" , chr correspond to "chr"+str (str)
#       -BF : Bayesian factor (float)
#       -reads_expected : average of reads covering the interval for all control samples (int)
#       -reads_observed : reads observed in the target patient for the interval 
#       -reads_ratio : reads_observed/reads_expected (float)
#   - A TSV file containing the reference samples lists used for each target sample.
#   It contains 3 untitled columns :
#       -First : target sample name (str)
#       -Second : reference samples list for autosomes, all sample names are joined by "," forming a large string.  
#       -Third : reference samples list for gonosomes, same format as second column.

# This python script is a wrapper of the calling R script. 
# It allows to display the R script usage and to check the input parameters.
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
Given a fragment count file on the canonical transcripts exonic intervals,for a minimum of 20 samples,
the CNVs can be called.
Call algorithm : beta-binomial. 
Segmentation : hidden Markov chain model.
Two results files are printed to stdout in TSV format:
    - The calls results have a format identical to the basic ExomeDepth format (eq:15 columns).
    - The reference sample lists for autosome and gonosome are keeped for each target sample (eq: 3 columns). 
If a pre-existing calls file and a pre-existing ref file produced by this program are provided (with --calls
and --refs ), the ref sets(New vs Old) are compared :
    -if same, target sample calls and ref lists are copied in the two new outputs.
    -if not the same, a new call is performed and a the new ref Sets is saved.
OPTIONS:
    --counts [str] : a fragment count TSV file (possibly gzipped). First 4 columns hold the exon definitions
     ("CHR","START","END","EXONID"), subsequent columns (one per sample) hold the counts.
    --outputcalls [str] : name of a new CNV calling TSV file (with path).
    --outputref [str] : name of a new TSV file (with path) for saved reference set used for each sample.
    --calls [str] optional : a pre-parsed call file (with path), old cnv call lines are completed or
                            re-parsed if one or more new patients are added.
    --refs [str] optional : a pre-parsed reference sample set TSV file (with path)
                            column 1: name of the sample to be processed,
                            column 2: comma-delimited autosomes ref sample names list,
                            column 3: comma-delimited gonosomes ref sample names list.\n"""

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
            if not os.path.isfile(countFile):
                sys.exit("ERROR : counts file "+countFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--outputcalls"):
            NEWcallFile=value
            if os.path.isfile(NEWcallFile):
                sys.exit("ERROR : new Callfile "+NEWcallFile+" already exist. Try "+scriptName+" --help.\n") 
        elif opt in ("--outputrefs"):
            NEWrefFile=value
            if os.path.isfile(NEWrefFile):
                sys.exit("ERROR : new Callfile "+NEWrefFile+" already exist. Try "+scriptName+" --help.\n")
        elif opt in ("--calls"):
            OLDcallFile=value
            if not os.path.isfile(OLDcallFile):
                sys.exit("ERROR : old counts file "+OLDcallFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--refs"):
            OLDrefFile=value
            if not os.path.isfile(OLDrefFile):
                sys.exit("ERROR : old ref file "+OLDrefFile+" doesn't exist. Try "+scriptName+" --help.\n")

        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    # B) Checking that the mandatory parameter is present
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
    # F) R script usage for calling
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