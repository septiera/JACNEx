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
import numpy as np # numpy arrays
import re
import time
import shutil
from multiprocessing import Pool #parallelize processes
import logging

####### MAGE-CNV modules
import mageCNV.bed
import mageCNV.countsFile
import mageCNV.countFragments


###############################################################################################
################################ Functions ####################################################
###############################################################################################
# mergeCounts
# fill sample column in countsArray with the corresponding 1D np.array (countsSample)
# counts : nd array with Fragment counts results [numberOfExons]x[numberOfSamples] [int]
# colSampleIndex : sample column index in counts
# sampleCount :  
def mergeCounts(counts, colSampleIndex, sampleCounts):
    for rowExonIndex in range(len(sampleCounts)):
        counts[rowExonIndex,colSampleIndex] = sampleCounts[rowExonIndex]

################################################################################################
######################################## Main ##################################################
################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want scriptName rather than 'root'
    logger = logging.getLogger(scriptName)
    
    ##########################################
    # parse user-provided arguments
    # mandatory args
    bams=""
    bamsFrom=""
    bedFile=""
    # optional args with default values
    padding=10
    maxGap=1000
    countsFile=""
    tmpDir="/tmp/"
    samtools="samtools"
    samThreads=10
    countJobs=3

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- padding).
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after
padding and sorting, subsequent columns (one per BAM) hold the counts.
If a pre-existing counts file produced by this program with the same BED is provided (with --counts),
counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).
ARGUMENTS:
   --bams [str]: comma-separated list of BAM files
   --bams-from [str]: text file listing BAM files, one per line
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column 
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --counts [str] optional: pre-existing counts file produced by this program, possibly gzipped,
           coounts for requested BAMs will be copied from this file if present
   --padding [int] : number of bps used to pad the exon coordinates, default : """+str(padding)+"""
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """+str(maxGap)+"""
   --tmp [str]: pre-existing dir for temp files, faster is better (eg tmpfs), default: """+tmpDir+"""
   --samtools [str]: samtools binary (with path if not in $PATH), default: """+str(samtools)+""""
   --samthreads [int]: number of threads for samtools, default: """+str(samThreads)+""""
   --jobs [int] : number of threads to allocate for counting step, default:"""+str(countJobs)+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","bams=","bams-from=","bed=","counts=","maxGap=","tmp=","threads=","jobs="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+". Try "+scriptName+" --help\n")

    for opt, value in opts:
        # sanity-check and store arguments
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--bams"):
            bams=value
            # bams is checked later, along with bamsFrom content
        elif opt in ("--bams-from"):
            bamsFrom=value
            if not os.path.isfile(bamsFrom):
                sys.exit("ERROR : bams-from file "+bamsFrom+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--bed"):
            bedFile=value
            if not os.path.isfile(bedFile):
                sys.exit("ERROR : bedFile "+bedFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--counts"):
            countsFile=value
            if not os.path.isfile(countsFile):
                sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--padding"):
            padding=int(value)
            if (padding<0):
                sys.exit("ERROR : padding "+str(padding)+" must be a positive int. Try "+scriptName+" --help.\n")
        elif opt in ("--maxGap"):
            maxGap=int(value)
            if (maxGap<0):
                sys.exit("ERROR : maxGap "+str(maxGap)+" must be a positive int. Try "+scriptName+" --help.\n")
        elif opt in ("--tmp"):
            tmpDir=value
            if not os.path.isdir(tmpDir):
                sys.exit("ERROR : tmp directory "+tmpDir+" doesn't exist. Try "+scriptName+" --help.\n")
        elif opt in ("--samtools"):
            samtools=value
            if shutil.which(samtools) is None:
                sys.exit("ERROR : samtools "+samtools+" cannot be run (wrong path, or binary not in $PATH?). Try "+scriptName+" --help.\n")
        elif opt in ("--samthreads"):
            samThreads=int(value)
            if (threads<=0):
                sys.exit("ERROR : threads "+str(threads)+" must be a positive int. Try "+scriptName+" --help.\n")
        elif opt in ("--jobs"):
            countJobs=int(value)
            if (countJobs<=0):
                sys.exit("ERROR : threads allocated for counting step "+str(countJobs)+" must be a positive int. Try "+scriptName+" --help.\n")      
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    # Check that the mandatory parameters are present
    if (bams=="" and bamsFrom=="") or (bams!="" and bamsFrom!=""):
        sys.exit("ERROR : You must use either --bams or --bams-from but not both.\n"+usage)
    if bedFile=="":
        sys.exit("ERROR : You must use --bed.\n"+usage)

    #####################################################
    # Check and clean up the provided list of BAMs
    # bamsTmp is user-supplied and may have dupes
    bamsTmp=[]
    # bamsNoDupe: tmp dictionary for removing dupes if any: key==bam, value==1
    bamsNoDupe={}
    # bamsToProcess, with any dupes removed
    bamsToProcess=[]
    # sample names stripped of path and .bam extension, same order as in bamsToProcess 
    samples=[]

    if bams != "":
        bamsTmp=bams.split(",")
    else:
        bamsList = open(bamsFrom,"r")
        for bam in bamsList:
            bam = bam.rstrip()
            bamsTmp.append(bam)

    # Check that all bams exist and remove any duplicates
    for bam in bamsTmp:
        if not os.path.isfile(bam):
            sys.exit("ERROR : BAM "+bam+" doesn't exist. Try "+scriptName+" --help.\n")
        elif bam in bamsNoDupe:
            logger.warning("BAM "+bam+" specified twice, ignoring the dupe")
        else:
            bamsNoDupe[bam]=1
            bamsToProcess.append(bam)
            sampleName=os.path.basename(bam)
            sampleName=re.sub(".bam$","",sampleName)
            samples.append(sampleName)

    ######################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    # parse exons from BED to obtain a list of lists (dim=NbExon x [CHR,START,END,EXONID]),
    # the exons are sorted according to their genomic position and padded
    try:
        exons = mageCNV.bed.processBed(bedFile, padding)
    except Exception:
        logger.error("processBed failed")
        sys.exit(1)
        
    thisTime = time.time()
    logger.debug("Done pre-processing BED, in %.2f s", thisTime-startTime)
    startTime = thisTime

    # allocate countsArray and countsFilled, and populate them with pre-calculated
    # counts if countsFile was provided.
    # countsArray[exonIndex,sampleIndex] will store the specified count,
    # countsFilled[sampleIndex] is True iff counts for specified sample were filled from countsFile
    try:
        (countsArray, countsFilled) =  mageCNV.countsFile.extractCountsFromPrev(exons, samples, countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed - %s", e)
        sys.exit(1)
    thisTime = time.time()
    logger.debug("Done parsing previous countsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime


    #####################################################
    # Process remaining (new) BAMs
    # data structure in the form of a queue where each result is stored 
    # if countFrags is completed (np array 1D counts stored for each sample)
    results = []

    # if countFrags is completed save the samples indexes
    processedBams = []

    # if countFrags fails for any BAMs, we have to remember their indexes
    # and only expunge them at the end, after exiting the for bamIndex loop
    # -> save their indexes in failedBams
    failedBams = []

    # The pool function allows to define the number of jobs to run.
    # WARNING : don't forget that the countFrags function parallels the samtools processes 
    # By default countsJobs = 3 jobs ; 3x10 threads = 30 cpu for samtools at the same times  
    with Pool(countJobs) as pool:
        for bamIndex in range(len(bamsToProcess)):
            bam = bamsToProcess[bamIndex]
            sample = samples[bamIndex]
            if countsFilled[bamIndex]:
                logger.info('Sample %s already filled from countsFile', sample)
                continue
            else:
                logger.info('Processing BAM for sample %s', sample)
                ####################
                # Fragment counting parallelization
                # apply module allows to set several arguments
                # async module allows not to block processes when finished
                # The output count np array results are placed in a queue which can then be 
                # retrieved by the get() command.
                # Note: that all bam's must be processed to retrieve the results.
                try:
                    results.append(pool.apply_async(mageCNV.countFragments.countFrags,args=(bam,exons,maxGap,tmpDir,samtools,samThreads)))
                    processedBams.append(bamIndex)
                
                # Raise an exception if counting error and storage the failed sample index in failedBams.
                except Exception as e:
                    logger.warning("Failed to count fragments for sample %s, skipping it - exception: %s",
                               sample, e)
                    failedBams.append(bamIndex)
        pool.close()
        pool.join()
        
    thisTime = time.time()
    logger.debug("Done processing all BAMs, in %.2f s", thisTime-startTime)
    startTime = thisTime

    # Copy sample results into local counts array
    for ListIndex in range(len(processedBams)):
        mergeCounts(countsArray, processedBams[ListIndex], results[ListIndex].get())
        
    # Expunge samples for which countFrags failed
    for failedI in reversed(failedBams):
        del(samples[failedI])
        countsArray = np.delete(countsArray,failedI,1)

    #####################################################
    # Print exon defs + counts to stdout
    mageCNV.countsFile.printCountsFile(exons, samples, countsArray)

    thisTime = time.time()
    logger.debug("Done merging and printing results for all samples, in %.2f s", thisTime-startTime)
    logger.info("ALL DONE")

if __name__ =='__main__':
    main()
