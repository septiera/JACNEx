import sys
import getopt
import glob
import os
import tempfile
import logging
from datetime import datetime


####### MAGE-CNV modules
import s1_countFrags
import s2_clusterSamps


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with:
# - everything needed by this module's main()
# - one sys.argv-like list (as a list of strings) for each mageCNV step
# If anything is wrong, raise Exception("EXPLICIT ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # args needed by main()
    workDir = ""

    # sys.argv-like lists for each mageCNV step
    step1Args = ["s1_countFrags.py"]
    step2Args = ["s2_clusterSamps.py"]
    step3Args = ["s3_CNCalls.py"]
    step4Args = ["s4_TBN.py"]

    # default values of global optional args, as strings
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))
    jobs = str(jobs)

    # default values of step1 optional args, as strings
    tmpDir = "/tmp/"
    padding = "10"
    maxGap = "1000"
    samtools = "samtools"

    # default values of step2 optional args, as strings
    minSamps = "20"
    maxCorr = "0.95"
    minCorr = "0.85"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
blablabla

ARGUMENTS:
Global arguments:
   --bams [str] : comma-separated list of BAM files (incompatible with --bams-from)
   --bams-from [str] : text file listing BAM files, one per line (incompatible with --bams)
   --bed [str] : BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --workDir [str] : subdir where intermediate results and QC files are produced, provide a pre-existing
           workDir to reuse results from a previous run (incremental use-case)
   --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + jobs + """
   -h , --help : display this help and exit

Step 1 optional arguments, defaults should be OK:
   --tmp [str] : pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + padding + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + maxGap + """
   --samtools [str] : samtools binary (with path if not in $PATH), default: """ + samtools + """

Step 2 optional arguments, defaults should be OK:
   --minSamps [int]: blablabla, default : """ + minSamps + """
   --maxCorr [float]: blablabla, default : """ + maxCorr + """
   --minCorr [float]: blablabla, default : """ + minCorr + """
   --noGender : blablabla
"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["bams=", "bams-from=", "bed=", "workDir=", "jobs=",
                                                       "help", "tmp=", "padding=", "maxGap=", "samtools=",
                                                       "minSamps=", "maxCorr=", "minCorr=", "noGender"])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--workDir"):
            workDir = value
        elif opt in ("--bams", "--bams-from", "--bed", "--tmp", "--padding", "--maxGap", "--samtools"):
            step1Args.extend([opt, value])
        elif opt in ("--jobs"):
            step1Args.extend([opt, value])
            step3Args.extend([opt, value])
        elif opt in ("--minSamps", "--maxCorr", "--minCorr"):
            step2Args.extend([opt, value])
        elif opt in ("--noGender"):
            step2Args.append(opt)
        else:
            raise Exception("unhandled option " + opt)

    # set default values if user didn't specify them
    if "--tmp" not in step1Args:
        step1Args.extend(["--tmp", tmpDir])
    if "--padding" not in step1Args:
        step1Args.extend(["--padding", padding])
    if "--maxGap" not in step1Args:
        step1Args.extend(["--maxGap", maxGap])
    if "--samtools" not in step1Args:
        step1Args.extend(["--samtools", samtools])
    if "--jobs" not in step1Args:
        step1Args.extend(["--jobs", jobs])

    if "--minSamps" not in step2Args:
        step2Args.extend(["--minSamps", minSamps])
    if "--maxCorr" not in step2Args:
        step2Args.extend(["--maxCorr", maxCorr])
    if "--minCorr" not in step2Args:
        step2Args.extend(["--minCorr", minCorr])

    if "--jobs" not in step3Args:
        step3Args.extend(["--jobs", jobs])

    #####################################################
    # process mageCNV.py-specific options, other options will be checked by s[1-4]_*.parseArgs()
    if workDir == "":
        raise Exception("you must provide a workDir with --workDir. Try " + scriptName + " --help")
    elif not os.path.isdir(workDir):
        try:
            os.mkdir(workDir)
        except Exception as e:
            raise Exception("workDir " + workDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(workDir, step1Args, step2Args, step3Args, step4Args)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # strings for each step, for log messages / exception names
    stepNames = ("STEP0 - CHECK/MAKE SUBDIR HIERARCHY -", "STEP1 - COUNT FRAGMENTS -",
                 "STEP2 - CLUSTER SAMPLES -", "STEP3 - CALL EXON-LEVEL CNs -",
                 "STEP4 - CALL CNVs -")

    logger.info("called with: " + " ".join(argv[1:]))
    # parse, check and preprocess arguments
    (workDir, step1Args, step2Args, step3Args, step4Args) = parseArgs(argv)

    ##################
    # hard-coded subdir hierarchy of workDir, created if needed

    # step1: countsFiles are saved (date-stamped and gzipped) in countsDir
    countsDir = workDir + '/CountFiles/'
    if not os.path.isdir(countsDir):
        try:
            os.mkdir(countsDir)
        except Exception:
            raise Exception(stepNames[0] + " countsDir " + countsDir + "doesn't exist and can't be mkdir'd")

    # step1: breakpoint results are saved in BPDir
    BPDir = workDir + '/BreakPoints/'
    if not os.path.isdir(BPDir):
        try:
            os.mkdir(BPDir)
        except Exception:
            raise Exception(stepNames[0] + " BPDir " + BPDir + " doesn't exist and can't be mkdir'd")

    # step2: clusterFiles are saved (date-stamped) in clustersDir
    clustersDir = workDir + '/ClusterFiles/'
    if not os.path.isdir(clustersDir):
        try:
            os.mkdir(clustersDir)
        except Exception:
            raise Exception(stepNames[0] + " clustersDir " + clustersDir + "doesn't exist and can't be mkdir'd")

    # step2: QC plots from step2 go in plotDir
    plotDir = workDir + '/QCPlots/'
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception:
            raise Exception(stepNames[0] + " plotDir " + plotDir + " doesn't exist and can't be mkdir'd")

    # shared date+time stamp, for new files
    dateStamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    ##################
    # check arguments of all steps before starting any actual work:
    # each step requires files from previous steps, but these don't exist yet
    # => make a bogus empty file, and remove it after checking args of all steps
    with tempfile.NamedTemporaryFile() as bogusFH:
        bogusFile = bogusFH.name

        #########
        # complement step1Args and check them
        step1Args.extend(["--BPDir", BPDir])

        # find and reuse most recent pre-existing countsFile, if any
        countsFilesAll = glob.glob(countsDir + '/countsFile_*.gz')
        countsFilePrev = max(countsFilesAll, default='', key=os.path.getctime)
        if countsFilePrev != '':
            logger.info("will reuse most recent countsFile: " + countsFilePrev)
            step1Args.extend(["--counts", countsFilePrev])

        # new countsFile to create
        countsFile = countsDir + '/countsFile_' + dateStamp + '.tsv.gz'
        if os.path.isfile(countsFile):
            raise Exception(stepNames[1] + " countsFile " + countsFile + " already exists")
        step1Args.extend(["--out", countsFile])

        # check step1 args, discarding results
        try:
            s1_countFrags.parseArgs(step1Args)
        except Exception as e:
            # problem is described in Exception, complement and reraise
            raise Exception(stepNames[1] + " parseArgs problem: " + str(e))

        #########
        # complement step2Args and check them
        step2Args.extend(["--plotDir", plotDir])

        # new clustersFile to create
        clustersFile = clustersDir + '/clustersFile_' + dateStamp + '.tsv'
        if os.path.isfile(clustersFile):
            raise Exception(stepNames[2] + " clustersFile " + clustersFile + " already exists")
        step2Args.extend(["--out", clustersFile])

        step2ArgsForCheck = step2Args.copy()
        step2ArgsForCheck.extend(["--counts", bogusFile])

        # check step2 args, discarding results
        try:
            s2_clusterSamps.parseArgs(step2ArgsForCheck)
        except Exception as e:
            # problem is described in Exception, complement and reraise
            raise Exception(stepNames[2] + " parseArgs problem: " + str(e))

        #########
        # complement step3Args and check them
        # TODO similarly to step2, using bogusFile for --counts and --clusters

    ######################
    logger.info("arguments look OK, starting to work")

    #########
    # step 1
    logger.info("%s STARTING", stepNames[1])
    try:
        s1_countFrags.main(step1Args)
    except Exception as e:
        logger.error("%s FAILED: %s", stepNames[1], str(e))
        raise Exception("STEP1 FAILED, check log")

    # countsFile wasn't created if it would be identical to countsFilePrev, if this
    # is the case just use countsFilePrev downstream
    if not os.path.isfile(countsFile):
        countsFile = countsFilePrev
    logger.info("%s DONE", stepNames[1])

    #########
    # step 2
    logger.info("%s STARTING", stepNames[2])
    step2Args.extend(["--counts", countsFile])
    try:
        s2_clusterSamps.main(step2Args)
    except Exception as e:
        logger.error("%s FAILED: %s", stepNames[2], str(e))
        raise Exception("STEP2 FAILED, check log")
    logger.info("%s DONE", stepNames[2])

    logger.info("ALL DONE")


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
