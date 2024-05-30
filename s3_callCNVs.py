###############################################################################################
######################## JACNEx step 3: exon filtering and calling ############################
###############################################################################################
# Given fragment counts produced by 1_countFrags.py and clusters of samples produced by
# 2_clusterSamps.py, call CNVs.
# See usage for details.
###############################################################################################
import getopt
import logging
import numpy
import os
import sys
import time

####### JACNEx modules
import countFrags.bed
import countFrags.countsFile
import clusterSamps.clustFile
import callCNVs.callsFile
import callCNVs.likelihoods
import callCNVs.priors
import callCNVs.transitions
import callCNVs.viterbi
import figures.plots

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
# If anything is wrong, raise Exception("EXPLICIT ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    clustsFile = ""
    outFileRoot = ""
    madeBy = ""
    # optional args with default values
    minGQ = 2.0
    padding = 10
    regionsToPlot = ""
    plotDir = ""
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Accepts exon fragment count data (from 1_countFrags.py) and sample clustering information
(from 2_clusterSamps.py) as input.
Performs several critical operations:
    a) Determines parameters for CN0 (half Gaussian) and CN2 (Gaussian) distributions for
       autosomal and gonosomal exons.
    b) Excludes non-interpretable exons based on set criteria.
    c) Calculates likelihoods for each CN state across exons and samples.
    d) Generates a transition matrix for CN state changes.
    e) Applies a Hidden Markov Model (HMM) to call and group CNVs.
    f) Outputs the CNV calls in VCF format.
The script utilizes multiprocessing for efficient computation and is structured to handle
errors and exceptions effectively, providing clear error messages for troubleshooting.
In addition, plots of FPMs and CN0-CN3+ models for specified samples+exons (if any) are
produced in plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 4 columns hold the sample cluster definitions.
                    [CLUSTER_ID, SAMPLES, FIT_WITH, VALID]. File obtained from 2_clusterSamps.py.
    --outFileRoot [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist.
    --minGQ [float]: minimum Genotype Quality score, default : """ + str(minGQ) + """
    --madeBy [str]: program name + version to print as "source=" in the produced VCF.
    --padding [int]: number of bps used to pad the exon coordinates, default : """ + str(padding) + """
    --regionsToPlot [str, optional]: comma-separated list of sampleID:chr:start-end for which exon-profile
               plots will be produced, eg "grex003:chr2:270000-290000,grex007:chrX:620000-660000"
    --plotDir [str]: subdir (created if needed) where exon-profile plots will be produced
    --jobs [int]: cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "clusts=", "outFileRoot=", "minGQ=",
                                                       "madeBy=", "padding=", "regionsToPlot=", "plotDir=", "jobs="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
        elif (opt in ("--clusts")):
            clustsFile = value
        elif opt in ("--outFileRoot"):
            outFileRoot = value
        elif opt in ("--minGQ"):
            minGQ = value
        elif opt in ("--madeBy"):
            madeBy = value
        elif opt in ("--padding"):
            padding = value
        elif (opt in ("--regionsToPlot")):
            regionsToPlot = value
        elif (opt in ("--plotDir")):
            plotDir = value
        elif opt in ("--jobs"):
            jobs = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    if clustsFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(clustsFile)):
        raise Exception("clustsFile " + clustsFile + " doesn't exist.")

    if outFileRoot == "":
        raise Exception("you must provide an outFileRoot with --outFileRoot. Try " + scriptName + " --help")
    elif (os.path.dirname(outFileRoot) != '') and (not os.path.isdir(os.path.dirname(outFileRoot))):
        raise Exception("the directory where outFileRoot " + outFileRoot + " should be created doesn't exist")

    if madeBy == "":
        raise Exception("you must provide a madeBy string with --madeBy. Try " + scriptName + " --help")

    #####################################################
    # Check other args
    try:
        minGQ = float(minGQ)
        if (minGQ <= 0):
            raise Exception()
    except Exception:
        raise Exception("minGQ must be a positive number, not " + str(minGQ))

    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    if regionsToPlot != "" and plotDir == "":
        raise Exception("you cannot provide --regionsToPlot without --plotDir")
    elif regionsToPlot == "" and plotDir != "":
        raise Exception("you cannot provide --plotDir without --regionsToPlot")
    elif regionsToPlot != "":
        # regionsToPlot: basic syntax check, discarding results;
        # if check fails, it raises an exception that just propagates to caller
        checkRegionsToPlot(regionsToPlot)

        # test plotdir last so we don't mkdir unless all other args are OK
        if not os.path.isdir(plotDir):
            try:
                os.mkdir(plotDir)
            except Exception as e:
                raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return (countsFile, clustsFile, outFileRoot, minGQ, padding, regionsToPlot, plotDir, jobs, madeBy)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (countsFile, clustsFile, outFileRoot, minGQ, padding, regionsToPlot, plotDir, jobs, madeBy) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.debug("starting to work")
    startTime = time.time()

    ###################
    # parse and FPM-normalize the counts, differentiating between exons on autosomes and gonosomes,
    # as well as intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.info("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # parse clusters informations and returns a tuple containing four dictionaries:
    # - clust2samps: cluster to samples mapping,
    # - samp2clusts: sample to clusters mapping,
    # - fitWith: cluster to similar clusters mapping,
    # - clust2gender: cluster gender (if gonosome),
    # - clustIsValid: cluster validity status.
    try:
        (clust2samps, samp2clusts, fitWith, clust2gender, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.info("Done parseClustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    clust2regions = preprocessRegionsToPlot(regionsToPlot, autosomeExons, gonosomeExons,
                                            samp2clusts, clustIsValid)

    ###################
    # call CNVs independently for each valid cluster
    for clusterID in sorted(clust2samps.keys()):
        if not clustIsValid[clusterID]:
            logger.info("cluster %s is INVALID, skipping it", clusterID)
            continue

        # VCF file for this cluster
        clustOutFile = outFileRoot + '_' + clusterID + '.vcf.gz'
        if os.path.exists(clustOutFile):
            raise Exception("outFile " + clustOutFile + " already exists")

        # for plotting we actually need:
        # key==exonIndex, value==list of lists[sampleIndex, sampleID]
        exonsToPlot = {}
        thisClust2reg = {}
        if clusterID in clust2regions:
            thisClust2reg = clust2regions[clusterID]

        # samplesInClust: temp dict, key==sampleID, value==1 if sample is in cluster
        # clusterID and value==2 if sample is in a FITWITH cluster for clusterID
        samplesInClust = {}
        for s in clust2samps[clusterID]:
            samplesInClust[s] = 1
        for fw in fitWith[clusterID]:
            for s in clust2samps[fw]:
                samplesInClust[s] = 2
        # OK we know how many samples are in clusterID + FITWITHs -> allocate
        # samplesOfInterest with True's, we will just need to set to FALSE
        # samples that belong to FITWITHs
        samplesOfInterest = numpy.ones(len(samplesInClust.keys()), dtype=bool)
        # sampleIndexes: 1D-array of indexes (in samples) of the sampleIDs that belong
        # to clusterID or its FITWITHs. Init to len(samples) for sanity-checking
        sampleIndexes = numpy.full(len(samplesInClust.keys()), len(samples), dtype=int)
        # clustSamples: list of sampleIDs in this cluster, in the same order as in samples
        clustSamples = []
        siClust = 0
        for si in range(len(samples)):
            thisSample = samples[si]
            if thisSample in samplesInClust:
                sampleIndexes[siClust] = si
                if samplesInClust[thisSample] == 2:
                    samplesOfInterest[siClust] = False
                else:
                    clustSamples.append(thisSample)
                if thisSample in thisClust2reg:
                    thisExon = thisClust2reg[thisSample]
                    if thisExon not in exonsToPlot:
                        exonsToPlot[thisExon] = []
                    exonsToPlot[thisExon].append([siClust, thisSample])
                siClust += 1

        # extract FPMs for the samples in cluster+FitWith (this actually makes a copy)
        clustIntergenicFPMs = intergenicFPMs[:, sampleIndexes]

        if clusterID.startswith("A_"):
            clustExonFPMs = autosomeFPMs[:, sampleIndexes]
            clustExons = autosomeExons
        else:
            clustExonFPMs = gonosomeFPMs[:, sampleIndexes]
            clustExons = gonosomeExons

        # by default, assume samples from this cluster are diploid for the chroms
        # that carry the clustExons
        isHaploid = False
        if (clusterID in clust2gender) and (clust2gender[clusterID] == 'M'):
            # Male => samples are haploid for for the sex chroms
            isHaploid = True
            # Females are diploid for chrX and don't have any chrY => NOOP

        callCNVsOneCluster(clustExonFPMs, clustIntergenicFPMs, samplesOfInterest, clustSamples,
                           clustExons, exonsToPlot, plotDir, clusterID, isHaploid, minGQ,
                           clustOutFile, padding, madeBy, jobs)

    thisTime = time.time()
    logger.info("all clusters done,  in %.1fs", thisTime - startTime)


###############################################################################
########################### PRIVATE FUNCTIONS #################################
###############################################################################
####################################################
# checkRegionsToPlot:
# do basic syntactic sanity-check of regionsToPlot, which should be a comma-separated
# list of sampleID:chr:start-end .
# If AOK, return a list of lists [str, str, int, int] holding [sampleID, chrom, start, end];
# else raise exception.
def checkRegionsToPlot(regionsToPlot):
    regions = []
    RTPs = regionsToPlot.split(',')
    for rtp in RTPs:
        rtpList = rtp.split(':')
        if len(rtpList) != 3:
            raise Exception("badly formatted regionToPlot, need 3 ':'-separated fields: " + rtp)
        startEnd = rtpList[2].split('-')
        if len(startEnd) != 2:
            raise Exception("badly formatted regionToPlot, need coords as start-end: " + rtp)
        (start, end) = startEnd
        try:
            start = int(start)
            end = int(end)
            if (start < 0) or (start > end):
                raise Exception()
        except Exception:
            raise Exception("badly formatted regionToPlot, must have 0 <= start <= end: " + rtp)
        regions.append([rtpList[0], rtpList[1], start, end])
    return(regions)


###################
# validate and pre-process each regionsToPlot:
# - does the sampleID exist? In what clusters?
# - does the chrom exist? In auto or gono?
# - are there any exons in the coords?
# If NO to any, log the issue and ignore this regionToPlot;
# if YES to all, populate and return clust2regions:
# key==clusterID, value==Dict with key==sampleID and value==list of exonIndexes
# (in the cluster's exons, auto or gono)
def preprocessRegionsToPlot(regionsToPlot, autosomeExons, gonosomeExons, samp2clusts, clustIsValid):
    clust2regions = {}
    if regionsToPlot == "":
        return(clust2regions)

    autosomeExonNCLs = countFrags.bed.buildExonNCLs(autosomeExons)
    gonosomeExonNCLs = countFrags.bed.buildExonNCLs(gonosomeExons)
    for region in checkRegionsToPlot(regionsToPlot):
        (sampleID, chrom, start, end) = region
        regionStr = sampleID + ':' + chrom + ':' + start + '-' + end
        if sampleID not in samp2clusts:
            logger.warning("ignoring bad regionToPlot %s, sample doesn't exist", regionStr)
            continue
        if chrom in autosomeExonNCLs:
            clustType = 'A_'
            exonNCLs = autosomeExonNCLs
        elif chrom in gonosomeExonNCLs:
            clustType = 'G_'
            exonNCLs = gonosomeExonNCLs
        else:
            logger.warning("ignoring bad regionToPlot %s, chrom doesn't exist", regionStr)
            continue

        clusterID = ""
        for clust in samp2clusts[sampleID]:
            if clust.startswith(clustType):
                clusterID = clust
                break

        if not clustIsValid[clusterID]:
            logger.warning("ignoring regionToPlot %s, sample belongs to invalid cluster %s",
                           regionStr, clusterID)
            continue

        overlappedExons = exonNCLs[chrom].find_overlap(start, end)
        if not overlappedExons:
            logger.warning("ignoring regionToPlot %s, region doesn't overlap any exons", regionStr)
            continue
        if clusterID not in clust2regions:
            clust2regions[clusterID] = {}
        if sampleID not in clust2regions[clusterID]:
            clust2regions[clusterID][sampleID] = []
        for exon in overlappedExons:
            exonIndex = exon[2]
            clust2regions[clusterID][sampleID].append(exonIndex)

    return(clust2regions)


####################################################
# callCNVsOneCluster:
# call CNVs for each sample in one cluster. Results are produced as a single
# VCF file for the cluster.
#
# Args:
# - exonFPMs: 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is the FPM
#   count for exon e in sample s (includes samples in FITWITH clusters, these are
#   used for fitting the CN2)
# - intergenicFPMs: 2D-array of floats of size nbIntergenics * nbSamples,
#   intergenicFPMs[i,s] is the FPM count for intergenic pseudo-exon i in sample s
# - samplesOfInterest: 1D-array of bools of size nbSamples, value==True iff the sample
#   is in the cluster of interest (vs being in a FITWITH cluster)
# - sampleIDs: list of nbSOIs sampleIDs (==strings), must be in the same order
#   as the corresponding samplesOfInterest columns in exonFPMs
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - exonsToPlot: Dict with key==exonIndex, value==list of lists[sampleIndex, sampleID] for
#   which we need to plot the FPMs and CN0-CN3+ models
# - plotDir: subdir where plots will be created (if any)
# - clusterID: string, for logging
# - isHaploid: bool, if True this cluster of samples is assumed to be haploid
#   for all chromosomes where the exons are located (eg chrX and chrY in men)
# - minGQ: float, minimum Genotype Quality (GQ)
# - vcfFile: name of VCF file to create
# - padding, madeBy: for printCallsFile
# - jobs: number of jobs for the parallelized steps (currently viterbiAllSamples())
#
# Produce vcfFile, return nothing.
def callCNVsOneCluster(exonFPMs, intergenicFPMs, samplesOfInterest, sampleIDs, exons,
                       exonsToPlot, plotDir, clusterID, isHaploid, minGQ, vcfFile,
                       padding, madeBy, jobs):
    logger.info("cluster %s - starting to work", clusterID)
    startTime = time.time()
    startTimeCluster = startTime

    # fit CN0 model using intergenic pseudo-exon FPMs for all samples (including
    # FITWITHs).
    # Currently CN0 is modeled with a half-normal distribution (parameter: CN0sigma).
    # Also returns fpmCn0, an FPM value up to which data looks like it (probably) comes
    # from CN0. This will be useful later for identifying NOCALL exons.
    (CN0sigma, fpmCn0) = callCNVs.likelihoods.fitCNO(intergenicFPMs)
    thisTime = time.time()
    logger.debug("cluster %s - done fitCN0 -> CN0sigma=%.2f fpmCn0=%.2f, in %.1fs",
                 clusterID, CN0sigma, fpmCn0, thisTime - startTime)
    startTime = thisTime

    # fit CN2 model for each exon using all samples in cluster (including FITWITHs)
    (Ecodes, CN2means, CN2sigmas) = callCNVs.likelihoods.fitCN2(exonFPMs, clusterID, fpmCn0, isHaploid)
    thisTime = time.time()
    logger.debug("cluster %s - done fitCN2 in %.1fs", clusterID, thisTime - startTime)
    startTime = thisTime

    # log stats with the percentages of exons in each QC class
    logExonStats(Ecodes, clusterID)

    # we only want to calculate likelihoods for the samples of interest (not the FITWITHs)
    # => create a view with all FPMs, then squash with the FPMs of SOIs if needed
    FPMsSOIs = exonFPMs
    if exonFPMs.shape[1] != samplesOfInterest.sum():
        FPMsSOIs = exonFPMs[:, samplesOfInterest]

    # use the fitted models to calculate likelihoods for all exons in all SOIs
    likelihoods = callCNVs.likelihoods.calcLikelihoods(FPMsSOIs, CN0sigma, Ecodes, CN2means, CN2sigmas, isHaploid)
    thisTime = time.time()
    logger.debug("cluster %s - done calcLikelihoods in %.1fs", clusterID, thisTime - startTime)
    startTime = thisTime

    # plot exonsToPlot if any
    figures.plots.plotExons(plotDir, exonsToPlot, FPMsSOIs, CN0sigma, Ecodes, CN2means, CN2sigmas, fpmCn0, isHaploid)

    # calculate priors (maxing the posterior probas iteratively until convergence)
    priors = callCNVs.priors.calcPriors(likelihoods)
    formattedPriors = "  ".join(["%.2e" % x for x in priors])
    logger.debug("cluster %s - priors = %s", clusterID, formattedPriors)
    thisTime = time.time()
    logger.debug("cluster %s - done calcPriors in %.1fs", clusterID, thisTime - startTime)
    startTime = thisTime

    # calculate metrics for building and adjusting the transition matrix, ignoring
    # NOCALL exons
    (baseTransMatMaxIED, adjustTransMatDMax) = countFrags.bed.calcIEDCutoffs(exons, Ecodes)

    # build matrix of base transition probas
    transMatrix = callCNVs.transitions.buildBaseTransMatrix(likelihoods, exons, priors, baseTransMatMaxIED)
    formattedMatrix = "\n\t".join(["\t".join([f"{cell:.2e}" for cell in row]) for row in transMatrix])
    logger.debug("cluster %s - base transition matrix =\n\t%s", clusterID, formattedMatrix)
    thisTime = time.time()
    logger.debug("cluster %s - done buildBaseTransMatrix in %.1fs", clusterID, thisTime - startTime)
    startTime = thisTime

    # call CNVs with the Viterbi algorithm
    CNVs = callCNVs.viterbi.viterbiAllSamples(likelihoods, sampleIDs, exons, transMatrix,
                                              priors, adjustTransMatDMax, minGQ, jobs)
    thisTime = time.time()
    logger.debug("cluster %s - done viterbiAllSamples in %.1fs", clusterID, thisTime - startTime)

    # set CN2means of NOCALL exons to 0
    CN2means[Ecodes < 0] = 0

    # print CNVs for this cluster as a VCF file
    callCNVs.callsFile.printCallsFile(vcfFile, CNVs, FPMsSOIs, CN2means, exons, sampleIDs,
                                      padding, madeBy)

    logger.info("cluster %s - all done, total time: %.1fs", clusterID, thisTime - startTimeCluster)
    return()


####################################################
# logExonStats:
# log stats with the percentages of exons in each QC class
def logExonStats(Ecodes, clusterID):
    # log exon statuses (as percentages)
    totalCnt = Ecodes.shape[0]
    # statusCnt: number of exons that passed (statusCnt[1]), semi-passed (statusCnt[0]), or
    # failed (statusCnt[2..5]) the QC criteria
    statusCnt = numpy.zeros(6, dtype=float)
    for s in range(6):
        statusCnt[s] = numpy.count_nonzero(Ecodes == 1 - s)
    statusCnt *= (100 / totalCnt)
    toPrint = "exon QC summary for cluster " + clusterID + ":\n\t"
    toPrint += "%.1f%% CALLED, %.1f%% CALLED-WITHOUT-CN1, " % (statusCnt[1], statusCnt[0])
    toPrint += "%.1f%% NOT-CAPTURED, %.1f%% FIT-CN2-FAILED, " % (statusCnt[2], statusCnt[3])
    toPrint += "%.1f%% CN2-LOW-SUPPORT, %.1f%% CN0-TOO-CLOSE" % (statusCnt[4], statusCnt[5])
    logger.info("%s", toPrint)


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
