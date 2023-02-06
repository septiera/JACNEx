###############################################################################################
######################################## MAGE-CNV step 1: count reads #########################
###############################################################################################
# Given a BED of exons and one or more BAM files, count the number of sequenced fragments
# from each BAM that overlap each exon (+- padding).
# See usage for details.
###############################################################################################
import sys
import getopt
import os
import math
import re
import time
import shutil
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor

####### MAGE-CNV modules
import countFrags.bed
import countFrags.countsFile
import countFrags.countFragments


# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


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

    # mandatory args
    bams = ""
    bamsFrom = ""
    bedFile = ""
    outFile = ""
    # optional args with default values
    BPDir = "BreakPoints/"
    countsFile = ""
    padding = 10
    maxGap = 1000
    tmpDir = "/tmp/"
    samtools = "samtools"
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- padding).
Results are printed to --out in TSV format (possibly gzipped): first 4 columns hold the exon
definitions after padding and sorting, subsequent columns (one per BAM) hold the counts.
If a pre-existing counts file produced by this program with the same BED is provided (with --counts),
counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).
In addition, any support for putative breakpoints is printed to sample-specific TSV files created in BPDir.

ARGUMENTS:
   --bams [str] : comma-separated list of BAM files
   --bams-from [str] : text file listing BAM files, one per line
   --bed [str] : BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --out [str] : file where results will be saved, must not pre-exist, will be gzipped if it ends with '.gz',
           can have a path component but the subdir must exist
   --BPDir [str] : dir (created if needed) where breakpoint files will be produced, default :  """ + BPDir + """
   --counts [str] optional: pre-existing counts file produced by this program, possibly gzipped,
           counts for requested BAMs will be copied from this file if present
   --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + str(maxGap) + """
   --tmp [str] : pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --samtools [str] : samtools binary (with path if not in $PATH), default: """ + samtools + """
   -h , --help : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "bams=", "bams-from=", "bed=", "out=", "BPDir=",
                                                       "counts=", "jobs=", "padding=", "maxGap=", "tmp=", "samtools="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            raise Exception(usage)
        elif opt in ("--bams"):
            bams = value
        elif opt in ("--bams-from"):
            bamsFrom = value
        elif opt in ("--bed"):
            bedFile = value
        elif opt in ("--out"):
            outFile = value
        elif opt in ("--BPDir"):
            BPDir = value
        elif opt in ("--counts"):
            countsFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif opt in ("--padding"):
            padding = value
        elif opt in ("--maxGap"):
            maxGap = value
        elif opt in ("--tmp"):
            tmpDir = value
        elif opt in ("--samtools"):
            samtools = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check and clean up list of BAMs
    if (bams == "" and bamsFrom == "") or (bams != "" and bamsFrom != ""):
        raise Exception("you must use either --bams or --bams-from but not both. Try " + scriptName + " --help")
    # bamsTmp will store user-supplied BAMs
    bamsTmp = []
    # bamsNoDupe: tmp dictionary for removing dupes if any: key==bam, value==1
    bamsNoDupe = {}
    # bamsToProcess, with any dupes removed
    bamsToProcess = []
    # sample names stripped of path and .bam extension, same order as in bamsToProcess
    samples = []

    if bams != "":
        bamsTmp = bams.split(",")
    elif not os.path.isfile(bamsFrom):
        raise Exception("bams-from file " + bamsFrom + " doesn't exist")
    else:
        try:
            bamsList = open(bamsFrom, "r")
            for bam in bamsList:
                bam = bam.rstrip()
                bamsTmp.append(bam)
            bamsList.close()
        except Exception as e:
            raise Exception("opening provided --bams-from file " + bamsFrom + " : " + str(e))

    # Check that all bams exist and remove any duplicates
    for bam in bamsTmp:
        if not os.path.isfile(bam):
            raise Exception("BAM " + bam + " doesn't exist")
        elif not os.access(bam, os.R_OK):
            raise Exception("BAM " + bam + " cannot be read")
        elif bam in bamsNoDupe:
            logger.warning("BAM " + bam + " specified twice, ignoring the dupe")
        else:
            bamsNoDupe[bam] = 1
            bamsToProcess.append(bam)
            sampleName = os.path.basename(bam)
            sampleName = re.sub(r"\.[bs]am$", "", sampleName)
            samples.append(sampleName)

    #####################################################
    # Check other args
    if bedFile == "":
        raise Exception("you must provide a BED file with --bed. Try " + scriptName + " --help")
    elif not os.path.isfile(bedFile):
        raise Exception("bedFile " + bedFile + " doesn't exist")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.isfile(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    if (countsFile != "") and (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist")

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))

    try:
        maxGap = int(maxGap)
        if (maxGap < 0):
            raise Exception()
    except Exception:
        raise Exception("maxGap must be a non-negative integer, not " + str(maxGap))

    if not os.path.isdir(tmpDir):
        raise Exception("tmp directory " + tmpDir + " doesn't exist")

    if shutil.which(samtools) is None:
        raise Exception("samtools program '" + samtools + "' cannot be run (wrong path, or binary not in $PATH?)")

    # test BPDir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(BPDir):
        try:
            os.mkdir(BPDir)
        except Exception:
            raise Exception("BPDir " + BPDir + " doesn't exist and can't be mkdir'd")

    # AOK, return everything that's needed
    return(bamsToProcess, samples, bedFile, outFile, BPDir, jobs, padding, maxGap, countsFile, tmpDir, samtools)


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
        (bamsToProcess, samples, bedFile, outFile, BPDir, jobs, padding, maxGap, countsFile, tmpDir, samtools) = parseArgs(argv)
    except Exception:
        # problem is described in Exception, just re-raise
        raise

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    # parse exons from BED to obtain a list of lists (dim=NbExon x [CHR,START,END,EXONID]),
    # the exons are sorted according to their genomic position and padded
    try:
        exons = countFrags.bed.processBed(bedFile, padding)
    except Exception:
        raise Exception("processBed failed")

    thisTime = time.time()
    logger.debug("Done pre-processing BED, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # allocate countsArray and countsFilled, and populate them with pre-calculated
    # counts if countsFile was provided.
    # countsArray[exonIndex,sampleIndex] will store the specified count,
    # countsFilled[sampleIndex] is True iff counts for specified sample were filled from countsFile
    try:
        (countsArray, countsFilled) = countFrags.countsFile.extractCountsFromPrev(exons, samples, countsFile)
    except Exception as e:
        raise Exception("parseCountsFile failed - " + str(e))

    thisTime = time.time()
    logger.debug("Done parsing previous countsFile, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # total number of samples that still need to be processed
    nbOfSamplesToProcess = len(bamsToProcess)
    for bamIndex in range(len(bamsToProcess)):
        if countsFilled[bamIndex]:
            nbOfSamplesToProcess -= 1

    # if bam2counts fails for any BAMs, we have to remember their indexes
    # and only expunge them at the end -> save their indexes in failedBams
    failedBams = []

    if nbOfSamplesToProcess == 0:
        logger.info("All requested samples are already in previous countsFile")

    else:
        #####################################################
        # decide how the work will be parallelized
        # we are allowed to use jobs cores in total: we will process paraSamples samples in
        # parallel, each sample will be processed using coresPerSample.
        # in our tests 7 cores/sample provided the best overall performance, both with jobs=20
        # and jobs=40. This probably depends on your hardware but in any case it's just a
        # performance tuning parameter.
        # -> we target targetCoresPerSample coresPerSample, this is increased if we
        #    have few samples to process (and we use ceil() so we may slighty overconsume)
        targetCoresPerSample = 7
        paraSamples = min(math.ceil(jobs / targetCoresPerSample), nbOfSamplesToProcess)
        coresPerSample = math.ceil(jobs / paraSamples)
        logger.info("%i new sample(s)  => will process %i in parallel, using up to %i cores/sample",
                    nbOfSamplesToProcess, paraSamples, coresPerSample)

        #####################################################
        # populate the module-global exonNCLs in countFragments
        try:
            countFrags.countFragments.initExonNCLs(exons)
        except Exception as e:
            raise Exception("initExonNCLs failed - " + str(e))

        #####################################################
        # Define nested callback for processing bam2counts() result (so countsArray et al
        # are in its scope)

        # mergeCounts:
        # arg: a Future object returned by ProcessPoolExecutor.submit(countFrags.countFragments.bam2counts).
        # bam2counts() returns a 3-element tuple (sampleIndex, sampleCounts, breakPoints).
        # If something went wrong, log and populate failedBams;
        # otherwise fill column at index sampleIndex in countsArray with counts stored in sampleCounts,
        # and print info about putative CNVs with alignment-supported breakpoints as TSV
        # to BPDir/sample.breakPoints.tsv
        def mergeCounts(futureBam2countsRes):
            e = futureBam2countsRes.exception()
            if e is not None:
                #  exceptions raised by bam2counts are always Exception(str(sampleIndex))
                si = int(str(e))
                logger.warning("Failed to count fragments for sample %s, skipping it", samples[si])
                failedBams.append(si)
            else:
                bam2countsRes = futureBam2countsRes.result()
                si = bam2countsRes[0]
                for exonIndex in range(len(bam2countsRes[1])):
                    countsArray[exonIndex, si] = bam2countsRes[1][exonIndex]
                if (len(bam2countsRes[2]) > 0):
                    try:
                        bpFile = BPDir + '/' + samples[si] + '.breakPoints.tsv'
                        BPFH = open(bpFile, mode='w')
                        for thisBP in bam2countsRes[2]:
                            toPrint = thisBP[0] + "\t" + str(thisBP[1]) + "\t" + str(thisBP[2]) + "\t" + thisBP[3] + "\t" + thisBP[4]
                            print(toPrint, file=BPFH)
                        BPFH.close()
                    except Exception as e:
                        logger.warning("Discarding breakpoints info for %s because cannot open %s for writing - %s",
                                       samples[si], bpFile, e)
                logger.info("Done counting fragments for %s", samples[si])

        #####################################################
        # Process new BAMs, up to paraSamples in parallel
        with ProcessPoolExecutor(paraSamples) as pool:
            for bamIndex in range(len(bamsToProcess)):
                bam = bamsToProcess[bamIndex]
                sample = samples[bamIndex]
                if countsFilled[bamIndex]:
                    logger.info('Sample %s already filled from countsFile', sample)
                    continue
                else:
                    futureRes = pool.submit(countFrags.countFragments.bam2counts,
                                            bam, len(exons), maxGap, tmpDir, samtools, coresPerSample, bamIndex)
                    futureRes.add_done_callback(mergeCounts)

        #####################################################
        # Expunge samples for which bam2counts failed
        if len(failedBams) > 0:
            for failedI in reversed(failedBams):
                del(samples[failedI])
            countsArray = np.delete(countsArray, failedBams, axis=1)

        thisTime = time.time()
        logger.info("Processed %i new BAMs in %.2fs, i.e. %.2fs per BAM",
                    nbOfSamplesToProcess, thisTime - startTime, (thisTime - startTime) / nbOfSamplesToProcess)
        startTime = thisTime

    #####################################################
    # Print exon defs + counts to outFile
    countFrags.countsFile.printCountsFile(exons, samples, countsArray, outFile)

    thisTime = time.time()
    logger.debug("Done printing counts for all (non-failed) samples, in %.2fs", thisTime - startTime)
    if len(failedBams) > 0:
        raise("counting FAILED for " + len(failedBams) + " samples, check the log!")
    else:
        logger.info("ALL DONE")


####################################################################################
######################################## Main ######################################
####################################################################################

if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))

    try:
        main(sys.argv)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + sys.argv[0] + " : " + str(e) + "\n")
        exit(1)
