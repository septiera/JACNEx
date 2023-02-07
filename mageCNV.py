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
# If anything is wrong, raise Exception("ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # global mandatory args
    bams = ""
    bamsFrom = ""
    bedFile = ""
    workDir = ""

    # global optional args
    tmpDir = "/tmp/"
    # jobs default: 80% of available cores, as a string
    jobs = round(0.8 * len(os.sched_getaffinity(0)))
    jobs = str(jobs)

    # step1 optional args with default values, as strings
    padding = "10"
    maxGap = "1000"
    samtools = "samtools"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
blablabla

ARGUMENTS:
Global arguments:
   --bams [str] : comma-separated list of BAM files
   --bams-from [str] : text file listing BAM files, one per line
   --bed [str] : BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --workDir [str] : subdir where intermediate results and QC files are produced, provide a pre-existing
           workDir to reuse results from a previous run (incremental use-case)
   --tmp [str] : pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + jobs + """
   -h , --help : display this help and exit

Step 1 optional arguments, defaults should be OK:
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + padding + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + maxGap + """
   --samtools [str] : samtools binary (with path if not in $PATH), default: """ + samtools + "\n"

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "bams=", "bams-from=", "bed=", "workDir=", "tmp=",
                                                       "jobs=", "padding=", "maxGap=", "samtools="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
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
            jobs = value
        elif opt in ("--padding"):
            padding = value
        elif opt in ("--maxGap"):
            maxGap = value
        elif opt in ("--samtools"):
            samtools = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # process mageCNV.py-specific options, other options will be checked by s[1-4]_*.py
    if workDir == "":
        raise Exception("you must provide a workDir with --workDir. Try " + scriptName + " --help")
    elif not os.path.isdir(workDir):
        try:
            os.mkdir(workDir)
        except Exception as e:
            raise Exception("workDir " + workDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(bams, bamsFrom, bedFile, workDir, tmpDir, jobs, padding, maxGap, samtools)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    try:
        (bams, bamsFrom, bedFile, workDir, tmpDir, jobs, padding, maxGap, samtools) = parseArgs(argv)
    except Exception:
        # problem is described in Exception, just re-raise
        raise

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")

    ######################
    # hard-coded subdir hierarchy of workDir, created if needed

    # name of current step, for log messages / exception names
    stepName = "STEP0 - CHECK SUBDIR HIERARCHY -"

    # countsFiles are saved (date-stamped and gzipped) in countsDir
    countsDir = workDir + '/CountFiles/'
    if not os.path.isdir(countsDir):
        try:
            os.mkdir(countsDir)
        except Exception:
            raise Exception(stepName + " countsDir " + countsDir + "doesn't exist and can't be mkdir'd")

    # breakpoint results are saved in BPDir
    BPDir = workDir + '/BreakPoints/'
    if not os.path.isdir(BPDir):
        try:
            os.mkdir(BPDir)
        except Exception:
            raise Exception(stepName + " BPDir " + BPDir + " doesn't exist and can't be mkdir'd")

    ######################
    # STEP 1
    stepName = "STEP1 - COUNT FRAGMENTS -"

    # build list of arguments for step1
    stepArgs = ["s1_countFrags.py"]
    stepArgs.extend(["--bams", bams, "--bams-from", bamsFrom, "--bed", bedFile])
    stepArgs.extend(["--tmp", tmpDir, "--jobs", jobs])
    stepArgs.extend(["--BPDir", BPDir, "--padding", padding, "--maxGap", maxGap, "--samtools", samtools])

    # find and reuse most recent pre-existing countsFile, if any
    countsFilesAll = glob.glob(countsDir + '/countsFile_*.gz')
    countsFilePrev = max(countsFilesAll, default='', key=os.path.getctime)
    if countsFilePrev != '':
        logger.info("will reuse most recent countsFile: " + countsFilePrev)
        stepArgs.extend(["--counts", countsFilePrev])

    # new countsFile to create: use date+time stamp
    dateStamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    countsFile = countsDir + '/countsFile_' + dateStamp + '.gz'
    if os.path.isfile(countsFile):
        raise Exception(stepName + " countsFile " + countsFile + " already exists")
    stepArgs.extend(["--out", countsFile])

    logger.info("%s STARTING", stepName)
    try:
        s1_countFrags.main(stepArgs)
    except Exception:
        logger.error("%s FAILED", stepName)
        raise

    # countsFile isn't created if it would be identical to countsFilePrev, if this
    # is the case just use countsFilePrev downstream
    if not os.path.isfile(countsFile):
        countsFile = countsFilePrev

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
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + sys.argv[0] + " : " + str(e) + "\n")
        sys.exit(1)
