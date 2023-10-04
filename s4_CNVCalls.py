###############################################################################################
################################ JACNEx  step 4:  CNV calling #################################
###############################################################################################
# Given three TSV files containing fragment counts, sample clusters, and distribution parameters
# for fitting the copy number profile.
# Generate a TSV file containing all the called CNVs for all samples.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import math
import time
import logging
import numpy as np
import concurrent.futures

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import clusterSamps.genderPrediction
import exonCalls.exonCallsFile
import CNVCalls.likelihoods
import CNVCalls.transitions
import CNVCalls.HMM

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS ################################
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
    countsFile = ""
    clustsFile = ""
    paramsFile = ""
    outFile = ""
    # optionnal args with default values
    jobs = round(0.8 * len(os.sched_getaffinity(0)))
    plotDir = "./plotDir/"

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Given three TSV files containing fragment counts, sample clusters, and distribution parameters
for fitting the copy number profile:
It calculates likelihoods for each sample and copy number, performs a hidden Markov chain
to obtain the best predictions, groups exons to form copy number variants (CNVs).
Generate a TSV file containing all the called CNVs for all samples.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
                    [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
                    File obtained from 2_clusterSamps.py.
    --params [str]: TSV file contains exon definitions in its first four columns,
                    followed by distribution parameters ["loc", "scale"] for exponential
                    and Gaussian distributions, and an additional column indicating the
                    exon filtering status for each cluster.
                    The file is generated using the 3_CNDistParams.py script.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
    --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
    --plotDir[str]: sub-directory in which the graphical PDFs will be produced, default:  """ + plotDir + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "params=", "out=",
                                                           "jobs=", "plotDir="])
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
        elif (opt in ("--params")):
            paramsFile = value
        elif opt in ("--out"):
            outFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif (opt in ("--plotDir")):
            plotDir = value
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

    if paramsFile == "":
        raise Exception("you must provide a continuous distribution parameters file use --params. Try " + scriptName + " --help.")
    elif (not os.path.isfile(paramsFile)):
        raise Exception("paramsFile " + paramsFile + " doesn't exist.")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other argsjobs = round(0.8 * len(os.sched_getaffinity(0)))
    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, clustsFile, paramsFile, outFile, jobs, plotDir)


###############################################################################
############################ PRIVATE FUNCTIONS #################################
###############################################################################
# exonOnChr:
# identifies the start and end indexes of "exons" for each chromosome.
#
# Arg:
# - exons (list of list[str, int, int, str]): containing CHR,START,END,EXON_ID
#
# Returns:
# - exonOnChrDict (dict): keys == chromosome identifiers, values == int list,
#                         ie [startChrExIndex, endChrExIndex]
#
def exonOnChr(exons):
    # Initialize an empty dictionary to store exons grouped by chromosome
    exonOnChrDict = {}

    prevChr = None
    for ei in range(len(exons)):
        currentChr = exons[ei][0]
        if currentChr == prevChr:
            continue
        else:
            exonOnChrDict[currentChr] = [ei]
            if prevChr is not None:
                exonOnChrDict[prevChr].append(ei - 1)

        prevChr = currentChr

    # For the last iteration, update the end index of the current chromosome
    exonOnChrDict[currentChr].append(ei)

    return exonOnChrDict


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
    (countsFile, clustsFile, paramsFile, outFile, jobs, plotDir) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse counts, perform FPM normalization, distinguish between intergenic regions and exons
    try:
        (samples, exons, intergenics, exonsFPM, intergenicsFPM) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # parse clusters informations
    try:
        (clust2samps, samp2clusts, fitWith, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # parse exon metrics for each valid cluster
    # extracts parameters of continuous distributions fitted on CN0, CN2 coverage profil
    try:
        (exMetrics, exp_loc, exp_scale, paramsTitles) = exonCalls.exonCallsFile.parseExonParamsFile(paramsFile)
    except Exception as e:
        raise Exception("parseParamsFile failed for %s : %s", paramsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing paramsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # Calling CNVs
    ####################
    # Entails a multi-step process to acquire data related to HMM parameters.
    # Defining unique variables to our model:
    # - States to be emitted by the HMM corresponding to different types of copy numbers:
    # CNO = homodeletion , CN1 = heterodeletion, CN2 = diploid (normal copy number)
    # CN3 = duplication, we decided not to distinguish the number of copies, so 3+.
    CNStates = ["CN0", "CN1", "CN2", "CN3"]
    # - CNState occurrence probabilities of the human genome, obtained from 1000 genome data
    # (doi:10.1186/1471-2105-13-305).
    priors = np.array([6.34e-4, 2.11e-3, 9.96e-1, 1.25e-3])

    #########
    # Likelihood calculation,
    # given count data (observation data in the HMM) and parameters of
    # continuous distributions computed for each cluster.
    # Concerns each hidden state(CNStates) and serve as a pseudo-emission
    # table(likelihoodsArray).
    # Cannot be performed within the Viterbi process itself, as it's
    # necessary for the generation of a transition matrix derived from
    # the data, which serves as one of the HMM parameters.

    # Allocates likelihoods dictionaries to store likelihood results specific to the
    # chromosome associated to each cluster.
    # keys= sampleID[str], values= likelihoodArray dim=[NBExons, [CN0;CN1,CN2,CN3+]]
    # "A"= autosomes, "G"=gonosomes
    likelihoods_A = {}
    likelihoods_G = {}

    # dictionary for easier retrieval of samplesID (key) and indexes in
    # the count table (values).
    samp2Index = {}
    for si in range(len(samples)):
        samp2Index[samples[si]] = si

    # This step is parallelized across clusters,
    paraClusters = min(math.ceil(jobs) // 3, len(clust2samps))
    logger.info("%i new clusters => will process %i in parallel", len(clust2samps), paraClusters)

    ##
    # mergeEmission:
    # arg: a Future object returned by ProcessPoolExecutor.submit(CNVCalls.likelihoods.counts2Likelihoods).
    # counts2Likelihoods returns a 3-element tuple (clusterID, chromType, likelihoodClustDict).
    # If something went wrong, raise error in log;
    # For each cluster, the chromType helps select the likelihoods dictionary to populate with the corresponding samples
    # and it's likelihoods arrays.
    # Some samples may be processed with autosomes and not with gonosomes (unvalid cluster), and vice versa.
    # The sampleIDs (keys) composition of the final dictionaries may not be identical.
    def mergeEmission(futurecounts2emission):
        e = futurecounts2emission.exception()
        if e is not None:
            clusterID = str(e)
            logger.warning("Failed counts2likelihoods for cluster %s, skipping it", clusterID)
        else:
            counts2emissionRes = futurecounts2emission.result()
            clusterID = counts2emissionRes[0]
            chromType = counts2emissionRes[1]

            if chromType == "A":
                likelihoods_A.update(counts2emissionRes[2])
            elif chromType == "G":
                likelihoods_G.update(counts2emissionRes[2])
            else:
                logger.error("chromType %s not implemented in mergeEmission", chromType)
                raise

            logger.debug("Likelihoods calculated for cluster %s", clusterID)

    # To be parallelised => browse clusters
    with concurrent.futures.ProcessPoolExecutor(paraClusters) as pool:
        for clusterID in clust2samps.keys():
            #### validity sanity check
            if not clustIsValid[clusterID]:
                logger.warning("Cluster %s is invalid, low sample number %i", clusterID, len(clust2samps[clusterID]))
                continue

            futureRes = pool.submit(CNVCalls.likelihoods.counts2likelihoods, clusterID, samp2Index, exonsFPM,
                                    clust2samps, exp_loc, exp_scale, exMetrics, len(CNStates))
            futureRes.add_done_callback(mergeEmission)

    thisTime = time.time()
    logger.debug("Done calculate likelihoods, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # uint8 numpy.ndarray of the same size as exons
    # 0=autosomes, 1=chrXZ, 2=chrYW
    exonOnSexChr = clusterSamps.genderPrediction.exonOnSexChr(exons)
    
    # dict: keys=chromosome identifiers[str], values=[startChrExIndex, endChrExIndex][int]
    chr2Exons = exonOnChr(exons)

    #########
    # Transition matrix generated from likelihood data, based on the overall sampling.
    # Contains an additional state, the 'void' state, it's a customization for the HMM
    # involves initializing and resetting HMM steps using priors.
    # The 'void' state does not appear among the emitted states.
    # np.ndarray 2D, dim = (nbOfCNStates + void) * (nbOfCNStates + void)
    try:
        plotFile = os.path.join(plotDir, "CN_Frequencies_Likelihoods_Plot.pdf")
        transMatrix = CNVCalls.transitions.getTransMatrix(likelihoods_A, likelihoods_G, chr2Exons,
                                                          priors, CNStates, samp2clusts, plotFile)
    except Exception as e:
        logger.error("getTransMatrix failed : %s", repr(e))
        raise Exception("getTransMatrix failed")

    thisTime = time.time()
    logger.debug("Done getTransMatrix, in %.2fs", thisTime - startTime)
    startTime = thisTime

    #########
    # Application of the HMM using the Viterbi algorithm.
    # returns a list of lists [CNtype, exonStart, exonEnd, sampleName].
    CNVs = []

    # this step is parallelized across samples.
    paraSample = min(math.ceil(jobs) // 3, len(samples))
    logger.info("%i samples => will process %i in parallel", len(samples), paraSample)

    ##
    # concatCNVs:
    # arg: a Future object returned by ProcessPoolExecutor.submit(groupCalls.HMM.viterbi).
    # viterbi returns a 4-element tuple (sampleID, CNVsList).
    # If something went wrong, raise error in log;
    # otherwise associating the sample name in the fourth list item and concatenate the
    # previous predictions(CNVs) with the new ones(CNVsList).
    def concatCNVs(futureViterbi):
        e = futureViterbi.exception()
        if e is not None:
            logger.warning("Failed viterbi for sample %s, skipping it", str(e))
        else:
            viterbiRes = futureViterbi.result()
            for sublist in viterbiRes[1]:
                sublist.append(viterbiRes[0])
            CNVs.extend(viterbiRes[1])

    # To be parallelised => browse samples
    with concurrent.futures.ProcessPoolExecutor(paraSample) as pool:
        for sampID in samples:

            for chrID in chr2Exons.keys():
                startChrExIndex = chr2Exons[chrID][0]
                endChrExIndex = chr2Exons[chrID][1]

                if exonOnSexChr[startChrExIndex] == 0:
                    if sampID in likelihoods_A:
                        CNcallOneSamp = likelihoods_A[sampID][startChrExIndex:endChrExIndex, :]
                    else:
                        continue
                else:
                    if sampID in likelihoods_G:
                        CNcallOneSamp = likelihoods_G[sampID][startChrExIndex:endChrExIndex, :]
                    else:
                        continue

                if np.all(CNcallOneSamp == -1):
                    continue

                futureRes = pool.submit(CNVCalls.HMM.viterbi, CNcallOneSamp, transMatrix, sampID, chrID)

                futureRes.add_done_callback(concatCNVs)

    thisTime = time.time()
    logger.debug("Done CNVs calls, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # ##########
    # # DEBUG PART
    # # Create a dictionary to count CNtypes per sampleName
    # count_dict = {}

    # # Iterate through the data list
    # for item in CNVs:
    #     # Destructure the elements of the list
    #     cn_type, _, _, sample_name = item
    #     if sample_name not in count_dict:
    #         # Initialize the counter for sampleName
    #         count_dict[sample_name] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    #     count_dict[sample_name][cn_type] += 1

    # # Log the dictionary using logger.debug()
    # for sample_name, counts in count_dict.items():
    #     # Format and log each row with keys and values aligned in columns
    #     logger.debug("%s: %d, %d, %d, %d, %d", sample_name, counts[0], counts[1], counts[2], counts[3], counts[4])

    ############
    # vcf format
    try:
        resVcf = CNVCalls.vcfFile.CNV2Vcf(CNVs, exons, samples, padding)
    except Exception as e:
        logger.error("CNV2Vcf failed : %s", repr(e))
        raise Exception("CNV2Vcf failed")

    thisTime = time.time()
    logger.debug("Done CNV2Vcf, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###########
    # print results
    CNVCalls.vcfFile.printVcf(resVcf, outFile, scriptName, samples)

    thisTime = time.time()
    logger.debug("Done printVcf, in %.2fs", thisTime - startTime)
    startTime = thisTime
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
