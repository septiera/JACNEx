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
import numpy as np
import re
import time
import shutil
from multiprocessing import Pool
import logging

####### MAGE-CNV modules
import mageCNV.bed
import mageCNV.countsFile
import mageCNV.countFragments


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
    outDir = "./breakPoints/"
    padding = 10
    maxGap = 1000
    countsFile = ""
    tmpDir = "/tmp/"
    samtools = "samtools"
    samThreads = 10
    countJobs = 3

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- padding).
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after
padding and sorting, subsequent columns (one per BAM) hold the counts.
If a pre-existing counts file produced by this program with the same BED is provided (with --counts),
counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).
In addition, any support for putative breakpoints is printed to sample-specific TSV files created in outDir.
ARGUMENTS:
   --bams [str]: comma-separated list of BAM files
   --bams-from [str]: text file listing BAM files, one per line
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --outDir [str]: subdir (created if needed) where breakpoint files will be produced, default :  """ + str(outDir) + """
   --counts [str] optional: pre-existing counts file produced by this program, possibly gzipped,
           coounts for requested BAMs will be copied from this file if present
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + str(maxGap) + """
   --tmp [str]: pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --samtools [str]: samtools binary (with path if not in $PATH), default: """ + str(samtools) + """"
   --samthreads [int]: number of threads for samtools, default: """ + str(samThreads) + """"
   --jobs [int] : number of threads to allocate for counting step, default:""" + str(countJobs) + "\n"

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "bams=", "bams-from=", "bed=", "outDir=", "counts=",
                                                       "padding=", "maxGap=", "tmp=", "samtools=", "samthreads=", "jobs="])
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
            # bams is checked later, along with bamsFrom content
        elif opt in ("--bams-from"):
            bamsFrom = value
            if not os.path.isfile(bamsFrom):
                sys.stderr.write("ERROR : bams-from file " + bamsFrom + " doesn't exist.\n")
                raise Exception()
        elif opt in ("--bed"):
            bedFile = value
            if not os.path.isfile(bedFile):
                sys.stderr.write("ERROR : bedFile " + bedFile + " doesn't exist.\n")
                raise Exception()
        elif opt in ("--outDir"):
            outDir = value
            if not os.path.isdir(outDir):
                try:
                    os.mkdir(outDir)
                except Exception:
                    sys.stderr.write("ERROR : outDir " + outDir + " doesn't exist and can't be mkdir'd\n")
                    raise Exception()
        elif opt in ("--counts"):
            countsFile = value
            if not os.path.isfile(countsFile):
                sys.stderr.write("ERROR : countsFile " + countsFile + " doesn't exist.\n")
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
        elif opt in ("--tmp"):
            tmpDir = value
            if not os.path.isdir(tmpDir):
                sys.stderr.write("ERROR : tmp directory " + tmpDir + " doesn't exist.\n")
                raise Exception()
        elif opt in ("--samtools"):
            samtools = value
            if shutil.which(samtools) is None:
                sys.stderr.write("ERROR : samtools program '" + samtools + "' cannot be run (wrong path, or binary not in $PATH?).\n")
                raise Exception()
        elif opt in ("--samthreads"):
            try:
                samThreads = int(value)
                if (samThreads <= 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : samthreads must be a positive integer, not '" + value + "'.\n")
                raise Exception()
        elif opt in ("--jobs"):
            try:
                countJobs = int(value)
                if (countJobs <= 0):
                    raise Exception()
            except Exception:
                sys.stderr.write("ERROR : jobs must be a positive integer, not '" + value + "'.\n")
                raise Exception()
        else:
            sys.stderr.write("ERROR : unhandled option " + opt + ".\n")
            raise Exception()

    #####################################################
    # Check that the mandatory parameters are present
    if (bams == "" and bamsFrom == "") or (bams != "" and bamsFrom != ""):
        sys.stderr.write("ERROR : You must use either --bams or --bams-from but not both. Try " + scriptName + " --help.\n")
        raise Exception()
    if bedFile == "":
        sys.stderr.write("ERROR : You must provide a BED file with --bed. Try " + scriptName + " --help.\n")
        raise Exception()

    # Check and clean up the provided list of BAMs
    # bamsTmp is user-supplied and may have dupes
    bamsTmp = []
    # bamsNoDupe: tmp dictionary for removing dupes if any: key==bam, value==1
    bamsNoDupe = {}
    # bamsToProcess, with any dupes removed
    bamsToProcess = []
    # sample names stripped of path and .bam extension, same order as in bamsToProcess
    samples = []

    if bams != "":
        bamsTmp = bams.split(",")
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
    # AOK, return everything that's needed
    return(bamsToProcess, samples, bedFile, outDir, padding, maxGap, countsFile, tmpDir, samtools, samThreads, countJobs)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, print error message to stderr and raise exception.
def main(argv):
    # parse, check and preprocess arguments - exceptions must be caught by caller
    (bamsToProcess, samples, bedFile, outDir, padding, maxGap, countsFile, tmpDir, samtools, samThreads, countJobs) = parseArgs(argv)

    # args seem OK, start working
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

    #####################################################
    # Define nested callback functions for apply_async (so countsArray et al
    # are in their scope)

    # mergeCounts:
    # can only accept one arg: the return value of mageCNV.countFragments.countFrags();
    # this must be a 3-element tuple (sampleIndex, sampleCounts, breakPoints).
    # Fill column at index sampleIndex in countsArray with counts stored in sampleCounts,
    # and print info about putative CNVs with alignment-supported breakpoints as TSV
    # to outDir/sample.breakPoints.tsv
    def mergeCounts(countFragsRes):
        for exonIndex in range(len(countFragsRes[1])):
            countsArray[exonIndex, countFragsRes[0]] = countFragsRes[1][exonIndex]
        if (len(countFragsRes[2]) > 0):
            # openOK is True iff we successfully opened bpFile for writing
            openOK = True
            try:
                bpFile = outDir + '/' + sample + '.breakPoints.tsv'
                BPFH = open(bpFile, mode='w')
            except Exception as e:
                logger.error("Cannot open breakpoints file %s for writing - %s", bpFile, e)
                logger.error("-> printing available breakpoint info for %s to stderr", sample)
                BPFH = sys.stderr
                openOK = False
            for thisBP in countFragsRes[2]:
                toPrint = thisBP[0] + "\t" + str(thisBP[1]) + "\t" + str(thisBP[2]) + "\t" + thisBP[3] + "\t" + thisBP[4]
                print(toPrint, file=BPFH)
            if openOK:
                BPFH.close()
            else:
                logger.error("Done printing breakpoint info for %s", sample)

    # error callback: if countFrags fails for any BAMs, we have to remember their indexes
    # and only expunge them at the end -> save their indexes in failedBams
    failedBams = []

    def jobError(e):
        logger.warning("Failed to count fragments for sample %s, skipping it - exception: %s", sample, e)
        failedBams.append(bamIndex)

    #####################################################
    # Process remaining (new) BAMs, up to countJobs in parallel
    # Careful: each BAM gets collated by samtools using samThreads threads before being processed
    # single-threaded by python code, so this can consume up to samThreads*countJobs cores
    with Pool(countJobs) as pool:
        for bamIndex in range(len(bamsToProcess)):
            bam = bamsToProcess[bamIndex]
            sample = samples[bamIndex]
            if countsFilled[bamIndex]:
                logger.info('Sample %s already filled from countsFile', sample)
                continue
            else:
                logger.info('Processing BAM for sample %s', sample)
                pool.apply_async(mageCNV.countFragments.countFrags,
                                 (bam, exons, maxGap, tmpDir, samtools, samThreads, bamIndex), {}, mergeCounts, jobError)
        pool.close()
        pool.join()

    thisTime = time.time()
    logger.debug("Done processing all BAMs, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # Expunge samples for which countFrags failed
    for failedI in reversed(failedBams):
        del(samples[failedI])
    countsArray = np.delete(countsArray, failedBams, axis=1)

    # Print exon defs + counts to stdout
    mageCNV.countsFile.printCountsFile(exons, samples, countsArray)

    thisTime = time.time()
    logger.debug("Done merging and printing results for all samples, in %.2f s", thisTime - startTime)
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
