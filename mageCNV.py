import sys
import getopt
import glob
import os
import logging
from datetime import datetime


####### MAGE-CNV modules
import s1_countFrags


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # global mandatory args
    bams = ""
    bamsFrom = ""
    bedFile = ""
    workDir = ""

    # global optional args
    tmpDir = "/tmp/"
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    # step1 optional args with default values
    padding = 10
    maxGap = 1000
    samtools = "samtools"

    usage = """\nCOMMAND SUMMARY:
blablabla
ARGUMENTS
Global arguments:
   --bams [str] : comma-separated list of BAM files
   --bams-from [str] : text file listing BAM files, one per line
   --bed [str] : BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --workDir [str] : subdir where intermediate results and QC files are produced, provide a pre-existing workDir to
           re-use results from a previous run (incremental use-case)
   --tmp [str] : pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
Step 1 optional arguments, defaults should be OK:
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + str(maxGap) + """
   --samtools [str] : samtools binary (with path if not in $PATH), default: """ + str(samtools) + """

   -h , --help : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "bams=", "bams-from=", "bed=", "workDir=", "tmp=",
                                                       "jobs=", "padding=", "maxGap=", "samtools="])
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : " + e.msg + ". Try " + scriptName + " --help\n")
        raise Exception()

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            raise Exception()
        elif opt in ("--bams"):
            bams = value
        elif opt in ("--bams-from"):
            bamsFrom = value
        elif opt in ("--bed"):
            bedFile = value
        elif opt in ("--workDir"):
            workDir = value
        elif opt in ("--tmp"):
            tmpDir = value
        elif opt in ("--jobs"):
            try:
                jobs = int(value)
                if (jobs <= 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : jobs must be a positive integer, not '" + value + "'.\n")
                raise Exception()
        elif opt in ("--padding"):
            try:
                padding = int(value)
                if (padding < 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : padding must be a non-negative integer, not '" + value + "'.\n")
                raise Exception()
        elif opt in ("--maxGap"):
            try:
                maxGap = int(value)
                if (maxGap < 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : maxGap must be a non-negative integer, not '" + value + "'.\n")
                raise Exception()
        elif opt in ("--samtools"):
            samtools = value
        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()

    #####################################################
    # process mageCNV.py-specific options, other options will be checked by s[1-4]_*.py
    if workDir == "":
        sys.stderr.write("ERROR : You must provide a workDir with --workDir. Try " + scriptName + " --help.\n")
        raise Exception()
    elif not os.path.isdir(workDir):
        try:
            os.mkdir(workDir)
        except Exception:
            sys.stderr.write("ERROR : workDir " + workDir + " doesn't exist and can't be mkdir'd\n")
            raise Exception()

    # AOK, return everything that's needed
    return(bams, bamsFrom, bedFile, workDir, tmpDir, jobs, padding, maxGap, samtools)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, log error message and raise exception.
def main(argv):
    # parse, check and preprocess arguments
    try:
        (bams, bamsFrom, bedFile, workDir, tmpDir, jobs, padding, maxGap, samtools) = parseArgs(argv)
    except Exception:
        # parseArgs explained on stderr what went wrong, just re-raise
        raise

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")

    ######################
    # hard-coded subdir hierarchy of workDir, created if needed

    # name of current step, for log messages
    stepName = "STEP0 - CHECK SUBDIR HIERARCHY -"

    # countsFiles are saved (date-stamped and gzipped) in countsDir
    countsDir = workDir + '/CountFiles/'
    if not os.path.isdir(countsDir):
        try:
            os.mkdir(countsDir)
        except Exception:
            logger.error("%s countsDir %s doesn't exist and can't be mkdir'd", stepName, countsDir)
            raise Exception()

    # breakpoint results are saved in BPDir
    BPDir = workDir + '/BreakPoints/'
    if not os.path.isdir(BPDir):
        try:
            os.mkdir(BPDir)
        except Exception:
            logger.error("%s BPDir %s doesn't exist and can't be mkdir'd", stepName, BPDir)
            raise Exception()

    ######################
    # STEP 1
    stepName = "STEP1 - COUNT FRAGMENTS -"

    # build list of arguments for step1
    stepArgs = ["s1_countFrags.py"]
    stepArgs.extend(["--bams", bams, "--bams-from", bamsFrom, "--bed", bedFile])
    stepArgs.extend(["--tmp", tmpDir, "--jobs", jobs])
    stepArgs.extend(["--BPDir", BPDir, "--padding", padding, "--maxGap", maxGap, "--samtools", samtools])

    # find and re-use most recent pre-existing countsFile, if any
    countsFilesAll = glob.glob(countsDir + '/countsFile_*.gz')
    countsFilePrev = max(countsFilesAll, default='', key=os.path.getctime)
    if countsFilePrev != '':
        stepArgs.extend(["--counts", countsFilePrev])

    # new countsFile to create: use date+time stamp
    dateStamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    countsFile = countsDir + '/countsFile_' + dateStamp + '.gz'
    if os.path.isfile(countsFile):
        logger.error("%s countsFile %s already exists\n", stepName, countsFile)
        raise Exception()
    stepArgs.extend(["--out", countsFile])

    logger.info("%s STARTING", stepName)
    try:
        s1_countFrags.main(stepArgs)
    except Exception:
        logger.error("%s FAILED", stepName)
        raise Exception()
    logger.info("%s DONE", stepName)


####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))

    try:
        main(sys.argv)
    except Exception as e:
        if (str(e) != ''):
            logger.error(str(e))
        exit(1)
