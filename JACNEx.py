import getopt
import glob
import gzip
import logging
import os
import re
import sys
import tempfile
from datetime import datetime


####### MAGE-CNV modules
import s1_countFrags
import s2_clusterSamps
import s3_callCNVs


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
    step3Args = ["s3_callCNVs.py"]

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

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
blablabla

ARGUMENTS:
Global arguments:
   --bams [str] : comma-separated list of BAM files (with path)
   --bams-from [str] : text file listing BAM files (with path), one per line
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
"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["bams=", "bams-from=", "bed=", "workDir=", "jobs=",
                                                       "help", "tmp=", "padding=", "maxGap=", "samtools=",
                                                       "minSamps="])
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
        elif opt in ("--minSamps"):
            step2Args.extend([opt, value])
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

    if "--padding" not in step3Args:
        step3Args.extend(["--padding", padding])
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
    return(workDir, step1Args, step2Args, step3Args)


####################################################
# Args:
# - countsFilesAll: list of pre-existing countsFiles (possibly empty), with PATH
# - samples: list of sample names
# Return (cf, commonSamples) where
# - cf is the countsFile with the most common samples, or '' if list was empty or
#   if no countsFile has any sample from samples
# - commonSamples is the number of common samples
def findBestPrevCF(countsFilesAll, samples):
    bestCF = ''
    commonSamples = 0
    # build dict of samples, value==1
    samplesD = {}
    for s in samples:
        samplesD[s] = 1
    for cf in countsFilesAll:
        try:
            if cf.endswith(".gz"):
                countsFH = gzip.open(cf, "rt")
            else:
                countsFH = open(cf, "r")
        except Exception as e:
            logger.error("Opening pre-existing countsFile %s: %s", cf, e)
            raise Exception('cannot open pre-existing countsFile')
        # grab samples from header
        samplesCF = countsFH.readline().rstrip().split("\t")
        # get rid of exon definition headers
        del samplesCF[0:4]
        commonSamplesCF = 0
        for s in samplesCF:
            if s in samplesD:
                commonSamplesCF += 1
        if commonSamplesCF > commonSamples:
            bestCF = cf
            commonSamples = commonSamplesCF
        countsFH.close()
    return(bestCF, commonSamples)


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
                 "STEP2 - CLUSTER SAMPLES -", "STEP3 - CALL CNVs -")

    # parse, check and preprocess arguments
    (workDir, step1Args, step2Args, step3Args) = parseArgs(argv)
    logger.info("called with: " + " ".join(argv[1:]))

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

    # step2: clusterFiles are saved (date-stamped and gzipped) in clustersDir
    clustersDir = workDir + '/ClusterFiles/'
    if not os.path.isdir(clustersDir):
        try:
            os.mkdir(clustersDir)
        except Exception:
            raise Exception(stepNames[0] + " clustersDir " + clustersDir + "doesn't exist and can't be mkdir'd")

    # step3: callsFiles are saved (date-stamped and gzipped) in callsDir
    callsDir = workDir + '/CallFiles/'
    if not os.path.isdir(callsDir):
        try:
            os.mkdir(callsDir)
        except Exception:
            raise Exception(stepNames[0] + " callsDir " + callsDir + "doesn't exist and can't be mkdir'd")

    # QC plots from step2 and step3 go in date-stamped subdirs of plotDir
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

        # not checking --counts here because we'll only know countsFilePrev later, but
        # we'll make sure findBestPrevCF() returns a file that exists

        # new countsFile to create
        countsFile = countsDir + '/countsFile_' + dateStamp + '.tsv.gz'
        if os.path.isfile(countsFile):
            raise Exception(stepNames[1] + " countsFile " + countsFile + " already exists")
        step1Args.extend(["--out", countsFile])

        # check step1 args, keeping only the list of samples
        try:
            samples = s1_countFrags.parseArgs(step1Args)[1]
        except Exception as e:
            # problem is described in Exception, complement and reraise
            raise Exception(stepNames[1] + " parseArgs problem: " + str(e))

        # find pre-existing countsFile (if any) with the most common samples
        countsFilesAll = glob.glob(countsDir + '/countsFile_*.gz')
        (countsFilePrev, commonSamples) = findBestPrevCF(countsFilesAll, samples)
        if commonSamples != 0:
            logger.info("will reuse best matching countsFile (%i common samples): %s",
                        commonSamples, countsFilePrev)
            step1Args.extend(["--counts", countsFilePrev])

        #########
        # complement step2Args and check them
        thisPlotDir = plotDir + 'QCPlots_' + dateStamp
        if os.path.isdir(thisPlotDir):
            raise Exception(stepNames[2] + " plotDir " + thisPlotDir + " already exists")
        step2Args.extend(["--plotDir", thisPlotDir])

        # new clustersFile to create
        clustersFile = clustersDir + '/clustersFile_' + dateStamp + '.tsv.gz'
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
        thisPlotDir = plotDir + 'exonFilteringPieCharts_' + dateStamp
        if os.path.isdir(thisPlotDir):
            raise Exception(stepNames[3] + " plotDir " + thisPlotDir + " already exists")
        step3Args.extend(["--plotDir", thisPlotDir])

        # new callsFile to create
        callsFile = callsDir + '/callsFile_' + dateStamp + '.vcf.gz'
        if os.path.isfile(callsFile):
            raise Exception(stepNames[3] + " callsFile " + callsFile + " already exists")
        step3Args.extend(["--out", callsFile])

        step3ArgsForCheck = step3Args.copy()
        step3ArgsForCheck.extend(["--counts", bogusFile])
        step3ArgsForCheck.extend(["--clusts", bogusFile])

        # check step3 args, discarding results
        try:
            s3_callCNVs.parseArgs(step3ArgsForCheck)
        except Exception as e:
            raise Exception(stepNames[3] + " parseArgs problem: " + str(e))

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
        if re.search(r'mismatched exons', str(e)):
            # specific exception string for this particular case
            raise Exception("STEP1 FAILED, use the same --bed and --padding as in previous runs or specify a new --workDir")
        else:
            raise Exception("STEP2 FAILED, check log")

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

    #########
    # step 3
    logger.info("%s STARTING", stepNames[3])
    step3Args.extend(["--counts", countsFile])
    step3Args.extend(["--clusts", clustersFile])
    try:
        s3_callCNVs.main(step3Args)
    except Exception as e:
        logger.error("%s FAILED: %s", stepNames[3], str(e))
        raise Exception("STEP3 FAILED, check log")
    logger.info("%s DONE", stepNames[3])

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
