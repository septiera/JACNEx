###############################################################################################
######################################## MAGE-CNV step 4:  call grouping     ##################
###############################################################################################
# Given a TSV of likelihoods per exon for 4 copy number states [CN0,CN1,CN2,CN3+] for each
# sample and TSV files of breakpoints for each sample, obtain a VCF file of copy number variations
# for all samples. See usage for more details.
###############################################################################################
import sys
import getopt
import os
import time
import logging

####### MAGE-CNV modules
import CNCalls.CNCallsFile

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
    callsFile = ""
    BPFolder = ""
    outFile = ""

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Given a TSV of likelihoods per exon for 4 copy number states [CN0,CN1,CN2,CN3+]
for each sample and TSV files of breakpoints for each sample.
Accurately identifies the CNs for each exon via the hidden markov chain (HMM)
algorithm by weighting the transitions with the lengths separating the exons
and by weighting the data-based transition matrix with the priors found in the literature.
Group exons to form CNVs, taking into account no-call exons.
Adjustment of precise delimitations if breakpoints have already been observed. 
Produces a VCF file of copy number variations for all samples.

ARGUMENTS:
    --calls [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the likelihoods per exon for 4 copy number states [CN0,CN1,CN2,CN3+]
            for each sample. File obtained from s3_CNCalls.py.
    --BPFolder [str]: folder containing gzipped or ungzipped TSV for all samples analysed.
                      Files obtained from s1_countFrags.py
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "calls=", "BPFolder=", "out="])
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
        elif (opt in ("--BPFolder")):
            BPFolder = value
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

    if BPFolder == "":
        raise Exception("you must provide a breakpoints folder use --BPFolder. Try " + scriptName + " --help.")
    elif (not os.path.isdir(BPFolder)):
        raise Exception("BreakPoints folder " + BPFolder + " doesn't exist.")

    #####################################################
    # Check other argsjobs = round(0.8 * len(os.sched_getaffinity(0)))
    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    # AOK, return everything that's needed
    return(callsFile , BPFolder, outFile)

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (callsFile , BPFolder, outFile) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()
    
    #############
    # parse calls
    try:
        CNStatus = ["CN0", "CN1", "CN2", "CN3"]
        (exons, samples, CNCallsArray) = CNCalls.CNCallsFile.parseCNcallsFile(callsFile, len(CNStatus))
    except Exception as e:
        logger.error("parseCNcallsFile failed for %s : %s", callsFile, repr(e))
        raise Exception("parseCNcallsFile failed")
    
    thisTime = time.time()
    logger.debug("Done parse callsFile, in %.2fs", thisTime - startTime)
    startTime = thisTime
    
    ############
    # 
    
    
    

####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    scriptName = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(scriptName)

    try:
        main(sys.argv)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + scriptName + " : " + str(e) + "\n")
        sys.exit(1)
