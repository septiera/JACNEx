###############################################################################################
######################################## MAGE-CNV step 1: count reads #########################
###############################################################################################
# Given a BED of exons and one or more BAM files, count the number of sequenced fragments
# from each BAM that overlap each exon (+- padding).
# Print results to stdout.
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
    bams = ""
    bamsFrom = ""
    bedFile = ""
    # optional args with default values
    BPdir = "./breakPoints/"
    countsFile = ""
    padding = 10
    maxGap = 1000
    tmpDir = "/tmp/"
    samtools = "samtools"
    jobs = 20

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- padding).
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after
padding and sorting, subsequent columns (one per BAM) hold the counts.
If a pre-existing counts file produced by this program with the same BED is provided (with --counts),
counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).
In addition, any support for putative breakpoints is printed to sample-specific TSV files created in BPdir.
ARGUMENTS:
   --bams [str]: comma-separated list of BAM files
   --bams-from [str]: text file listing BAM files, one per line
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --BPdir [str]: subdir (created if needed) where breakpoint files will be produced, default :  """ + str(BPdir) + """
   --counts [str] optional: pre-existing counts file produced by this program, possibly gzipped,
           counts for requested BAMs will be copied from this file if present
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + str(maxGap) + """
   --tmp [str]: pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --samtools [str]: samtools binary (with path if not in $PATH), default: """ + str(samtools) + """
   --jobs [int] : approximate number of cores that we can use, default:""" + str(jobs) + "\n" + """
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "bams=", "bams-from=", "bed=", "BPdir=", "counts=",
                                                       "padding=", "maxGap=", "tmp=", "samtools=", "jobs="])
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
        elif opt in ("--BPdir"):
            BPdir = value
        elif opt in ("--counts"):
            countsFile = value
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
        elif opt in ("--tmp"):
            tmpDir = value
        elif opt in ("--samtools"):
            samtools = value
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
    # Check and clean up list of BAMs
    if (bams == "" and bamsFrom == "") or (bams != "" and bamsFrom != ""):
        sys.stderr.write("ERROR : You must use either --bams or --bams-from but not both. Try " + scriptName + " --help.\n")
        raise Exception()
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
        sys.stderr.write("ERROR : bams-from file " + bamsFrom + " doesn't exist.\n")
        raise Exception()
    else:
        try:
            bamsList = open(bamsFrom, "r")
            for bam in bamsList:
                bam = bam.rstrip()
                bamsTmp.append(bam)
        except Exception as e:
            sys.stderr.write("ERROR opening provided --bams-from file %s: %s", bamsFrom, e)
            raise Exception()

    # Check that all bams exist and remove any duplicates
    for bam in bamsTmp:
        if not os.path.isfile(bam):
            sys.stderr.write("ERROR : BAM " + bam + " doesn't exist.\n")
            raise Exception()
        elif not os.access(bam, os.R_OK):
            sys.stderr.write("ERROR : BAM " + bam + " cannot be read.\n")
            raise Exception()
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
        sys.stderr.write("ERROR : You must provide a BED file with --bed. Try " + scriptName + " --help.\n")
        raise Exception()
    elif not os.path.isfile(bedFile):
        sys.stderr.write("ERROR : bedFile " + bedFile + " doesn't exist.\n")
        raise Exception()

    if (countsFile != "") and (not os.path.isfile(countsFile)):
        sys.stderr.write("ERROR : countsFile " + countsFile + " doesn't exist.\n")
        raise Exception()

    if not os.path.isdir(tmpDir):
        sys.stderr.write("ERROR : tmp directory " + tmpDir + " doesn't exist.\n")
        raise Exception()

    if shutil.which(samtools) is None:
        sys.stderr.write("ERROR : samtools program '" + samtools + "' cannot be run (wrong path, or binary not in $PATH?).\n")
        raise Exception()

    # test BPdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(BPdir):
        try:
            os.mkdir(BPdir)
        except Exception:
            sys.stderr.write("ERROR : BPdir " + BPdir + " doesn't exist and can't be mkdir'd\n")
            raise Exception()

    # AOK, return everything that's needed
    return(bamsToProcess, samples, bedFile, BPdir, padding, maxGap, countsFile, tmpDir, samtools, jobs)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (bamsToProcess, samples, bedFile, BPdir, padding, maxGap, countsFile, tmpDir, samtools, jobs) = parseArgs(argv)

    # args seem OK, start working
    logger.info("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    # parse exons from BED to obtain a list of lists (dim=NbExon x [CHR,START,END,EXONID]),
    # the exons are sorted according to their genomic position and padded
    try:
        exons = mageCNV.bed.processBed(bedFile, padding)
    except Exception:
        logger.error("processBed failed")
        raise Exception()

    thisTime = time.time()
    logger.debug("Done pre-processing BED, in %.2f s", thisTime - startTime)
    startTime = thisTime

    # allocate countsArray and countsFilled, and populate them with pre-calculated
    # counts if countsFile was provided.
    # countsArray[exonIndex,sampleIndex] will store the specified count,
    # countsFilled[sampleIndex] is True iff counts for specified sample were filled from countsFile
    try:
        (countsArray, countsFilled) = mageCNV.countsFile.extractCountsFromPrev(exons, samples, countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed - %s", e)
        raise Exception()

    thisTime = time.time()
    logger.debug("Done parsing previous countsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    # populate the module-global exonNCLs in countFragments
    try:
        mageCNV.countFragments.initExonNCLs(exons)
    except Exception as e:
        logger.error("initExonNCLs failed - %s", e)
        raise Exception()

    #####################################################
    # decide how the work will be parallelized

    # total number of samples that still need to be processed
    nbOfSamplesToProcess = len(bamsToProcess)
    for bamIndex in range(len(bamsToProcess)):
        if countsFilled[bamIndex]:
            nbOfSamplesToProcess -= 1

    # we are allowed to use jobs cores in total: we will process paraSamples samples in
    # parallel, each sample will be processed using coresPerSample.
    # in our tests countFrags scales great up to 4-5 coresPerSample, with diminishing
    # returns beyond that (probably depends on your hardware)
    # -> we target targetCoresPerSample coresPerSample, this is increased if we
    #    have few samples to process (and we use ceil() so we may slighty overconsume)
    targetCoresPerSample = 4
    paraSamples = min(math.ceil(jobs / targetCoresPerSample), nbOfSamplesToProcess)
    coresPerSample = math.ceil(jobs / paraSamples)
    logger.info("we will process %i samples in parallel, using up to %i cores for each sample.",
                paraSamples, coresPerSample)

    #####################################################
    # Define nested callback for processing countFrags() result (so countsArray et al
    # are in its scope)

    # if countFrags fails for any BAMs, we have to remember their indexes
    # and only expunge them at the end -> save their indexes in failedBams
    failedBams = []

    # mergeCounts:
    # arg: a Future object returned by ProcessPoolExecutor.submit(mageCNV.countFragments.countFrags).
    # countFrags() returns a 3-element tuple (sampleIndex, sampleCounts, breakPoints).
    # If something went wrong, log and populate failedBams;
    # otherwise fill column at index sampleIndex in countsArray with counts stored in sampleCounts,
    # and print info about putative CNVs with alignment-supported breakpoints as TSV
    # to BPdir/sample.breakPoints.tsv
    def mergeCounts(futureCountFragsRes):
        e = futureCountFragsRes.exception()
        if e is not None:
            #  exceptions raised by countFrags are always Exception(str(sampleIndex))
            si = int(str(e))
            logger.warning("Failed to count fragments for sample %s, skipping it", samples[si])
            failedBams.append(si)
        else:
            countFragsRes = futureCountFragsRes.result()
            si = countFragsRes[0]
            for exonIndex in range(len(countFragsRes[1])):
                countsArray[exonIndex, si] = countFragsRes[1][exonIndex]
            if (len(countFragsRes[2]) > 0):
                try:
                    bpFile = BPdir + '/' + samples[si] + '.breakPoints.tsv'
                    BPFH = open(bpFile, mode='w')
                    for thisBP in countFragsRes[2]:
                        toPrint = thisBP[0] + "\t" + str(thisBP[1]) + "\t" + str(thisBP[2]) + "\t" + thisBP[3] + "\t" + thisBP[4]
                        print(toPrint, file=BPFH)
                    BPFH.close()
                except Exception as e:
                    logger.warning("Discarding breakpoints info for %s because cannot open %s for writing - %s", samples[si], bpFile, e)
            logger.info("Done processing %s", samples[si])

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
                futureRes = pool.submit(mageCNV.countFragments.countFrags,
                                        bam, len(exons), maxGap, tmpDir, samtools, coresPerSample, bamIndex)
                futureRes.add_done_callback(mergeCounts)

    thisTime = time.time()
    logger.debug("Done processing all BAMs, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Expunge samples for which countFrags failed
    failedBamsNb = len(failedBams)
    for failedI in reversed(failedBams):
        del(samples[failedI])
    countsArray = np.delete(countsArray, failedBams, axis=1)

    # Print exon defs + counts to stdout
    mageCNV.countsFile.printCountsFile(exons, samples, countsArray)

    thisTime = time.time()
    logger.debug("Done printing results for all samples, in %.2f s", thisTime - startTime)
    if (failedBamsNb > 0):
        logger.warning("ALL DONE BUT COUNTING FAILED FOR %i SAMPLES, check the log!", failedBamsNb)
    else:
        logger.info("ALL DONE")


####################################################################################
######################################## Main ######################################
####################################################################################

if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))

    try:
        main(sys.argv)
    except Exception:
        # whoever raised the exception should have explained it on stderr, here we just die
        exit(1)
