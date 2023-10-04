###############################################################################################
################################ JACNEx  step 5:  VCF #########################################
###############################################################################################
# From the output of step 4, which comprises a list of lists containing called CNVs informations
# and sample-specific breakpoint files, generate a Variant Call Format (VCF) file.
# See usage for more details.
###############################################################################################

import sys
import getopt
import os
import math
import time
import logging
import numpy as np
import concurrent.futures

####### JACNEx modules
import CNVCalls.vcfFile

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)

###############################################################################
############################ PUBLIC FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
# If anything is wrong, raise Exception("ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    callsFile = sys.stdin
    BPDir = ""  # not optional as it is automatically generated
    # optionnal args with default values
    padding = 10
    

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Starting with a TSV file that contains Copy Number Variation (CNV) calls in the format
[CHR, START, END, CNTYPE, QUALITYSCORE, SAMPLEID], which was generated as a result of step 3,
and utilizing TSV files that contain patient-specific breakpoint information in the format
[CHR, START, END, CNVTYPE,	COUNT-QNAMES, QNAMES] obtained from step 1, the objective is to
carry out two primary tasks.
First, it involves the correction of exon-specific breakpoint positions, using the actual
breakpoint positions if they are available and corrects the padding.
Second, it requires formatting the data into the Variant Call Format (VCF).
ARGUMENTS:
    --calls [str]: tsv file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --BPDir [str]: folder containing gzipped or ungzipped TSV for all samples analysed.
                Files obtained from s1_countFrags.py
    --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "calls=", "BPDir=", "padding=", "out="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--calls")):
            callsFile = value
        elif (opt in ("--BPDir")):
            BPDir = value
        elif (opt in ("--padding")):
            padding = value
        elif opt in ("--out"):
            outFile = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if callsFile == "":
        raise Exception("you must provide a calls file with --calls. Try " + scriptName + " --help.")
    elif (not os.path.isfile(callsFile)):
        raise Exception("callsFile " + callsFile + " doesn't exist.")

    if BPDir == "":
        raise Exception("you must provide a breakpoint files directory use --BPDir. Try " + scriptName + " --help.")
    elif (not os.path.isdir(BPDir)):
        raise Exception("BPDir " + BPDir + " doesn't exist.")

    #####################################################
    # Check other args
    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    # AOK, return everything that's needed
    return(callsFile, BPDir, padding, outFile)

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (callsFile, BPDir, padding, outFile) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()
    
    # parse callsFile
    
    # parse BPdir
    
    # correct breakpoint locations
    
    # merge and format vcf

