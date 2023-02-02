import sys
import getopt
import os
import logging


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

    # mandatory args

    # optional args with default values
    tmpDir = "/tmp/"
    jobs = 20

    usage = """\nCOMMAND SUMMARY:
blablabla
ARGUMENTS:
   --tmp [str]: pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --jobs [int] : approximate number of cores that we can use, default:""" + str(jobs) + "\n" + """
    -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "tmp=", "jobs="])
    except getopt.GetoptError as e:
        sys.stderr.write("ERROR : " + e.msg + ". Try " + scriptName + " --help\n")
        raise Exception()

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            raise Exception()
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
        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()

    #####################################################
    # Check args
    if not os.path.isdir(tmpDir):
        sys.stderr.write("ERROR : tmp directory " + tmpDir + " doesn't exist.\n")
        raise Exception()

    # AOK, return everything that's needed
    return(tmpDir, jobs)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (tmpDir, jobs) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")

    # name of current step, for log messages
    stepName = "STEP1 - COUNT FRAGMENTS -"

    # build list of arguments for step1
    argsCount = ["s1_countFrags.py"]
    # JUST FOR TESTING: testing with hard-coded args
    argsCount.extend(["--bams", "grexome0561_smaller.bam", "--bed", "canonicalTranscripts_221021.bed.gz"])

    logger.info("%s STARTING", stepName)
    try:
        s1_countFrags.main(argsCount)
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
