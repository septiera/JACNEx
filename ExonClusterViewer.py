###############################################################################################
##################################### ExonClustViewer #########################################
###############################################################################################
# uses fragment count data, sample clustering information, and calculated parameters from
# various continuous distributions to generate coverage profiles.
# This tool enables the visualization of coverage profiles for a specific exon ID and
# a given sample given by a user TSV file.
# The resulting plots are saved in PDF format for further analysis and sharing.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import gzip
import numpy as np
import time
import logging
import scipy
import matplotlib.backends.backend_pdf

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import clusterSamps.genderPrediction
import exonCalls.exonCallsFile
import exonCalls.exonProcessing
import CNVCalls.likelihoods
import figures.plots

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
# If anything is wrong, raise Exception("EXPLICIT ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    clustsFile = ""
    tocheckFile = ""
    pdfFile = ""

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Given  four TSV files: one for fragment count data, another for sample cluster information,
a third for fitted continuous distribution parameters, and a user-defined file containing
two columns [exonID, sampleID] tab-separated.
For each exon, it identifies the associated filtering criteria.
If the filtering value is 0 ("notCaptured") or 1 ("cannotFitRG"), a simple histogram will be plotted.
In cases where filtering is 2 ("RGClose2LowThreshold"), 3 ("fewSampsInRG"), or 4 ("call"),
specific distributions will be fitted for each copy number (CN0: exponential, CN1: Gaussian,
CN2: Gaussian, CN3: gamma) based on the FPM (fragments per million) distribution for the cluster
containing the target sample.
The resulting visualization includes a combined histogram and distribution plot with their respective parameters.
All graphical representations will be consolidated and stored as a single output PDF file.

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
    --tocheck [str]: user-generated TSV file containing two columns [exonID(str), sampleID(str)] separated by tabs.
    --pdf[str]: a pdf file in which the graphical output will be produced.
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "params=",
                                                           "tocheck=", "pdf="])
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
        elif (opt in ("--tocheck")):
            tocheckFile = value
        elif (opt in ("--pdf")):
            pdfFile = value
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

    if tocheckFile == "":
        raise Exception("you must provide a user file use --tocheck. Try " + scriptName + " --help.")
    elif (not os.path.isfile(tocheckFile)):
        raise Exception("tocheckFile " + tocheckFile + " doesn't exist.")

    if pdfFile == "":
        raise Exception("you must provide an pdfFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(pdfFile):
        raise Exception("pdfFile " + pdfFile + " already exists")
    elif (os.path.dirname(pdfFile) != '') and (not os.path.isdir(os.path.dirname(pdfFile))):
        raise Exception("the directory where pdfFile " + pdfFile + " should be created doesn't exist")

    # AOK, return everything that's needed
    return(countsFile, clustsFile, paramsFile, tocheckFile, pdfFile)


#########################
# parseUserTSV
# Given a user-provided TSV file containing sample names and exon identifiers,
# along with a list of sample names and a list of exon definitions.
# It parses the TSV file to generate a list of exons to check for each sample.
# The function returns a list where each index corresponds to a sample and contains
# the indexes of matching exons from the exon definitions.
#
# Args:
# - userTSV (str): Path to the user TSV file.
# - samples (list[str]): List of sample names.
# - exons (list of lists[str, int, int, str]): List of exon definitions [CHR, START, END, EXONID].
#
# Returns:
# - samps2Check (list of lists[int]): List of exons to check for each sample.
#                                     Each sampleID corresponds to a list of matching "exons" indexes.
def parseUserTSV(userTSV, samples, exons):
    try:
        if userTSV.endswith(".gz"):
            userTSVFH = gzip.open(userTSV, "rt")
        else:
            userTSVFH = open(userTSV, "r")
    except Exception as e:
        logger.error("Opening provided userTSV %s: %s", userTSV, e)
        raise Exception('cannot open userTSV')

    samps2Check = {key: [] for key in samples}

    for line in userTSVFH:
        splitLine = line.rstrip().split("\t", maxsplit=1)
        # Check if the sample name is in the provided sample name list
        if splitLine[0] in samples:
            exonIDFound = False
            for exonID in range(len(exons)):
                # Check if the exon identifier is present in "exons"
                if exons[exonID][3] == splitLine[1]:
                    samps2Check[splitLine[0]].append(exonID)
                    exonIDFound = True

            if not exonIDFound:
                logger.error("Exon identifier %s has no match with the bed file provided in step 1.", splitLine[1])
                raise Exception("Exon identifier not found.")
        else:
            logger.error("Sample name %s has no match with the sample name list.", splitLine[0])
            raise Exception("Sample name not found.")

    userTSVFH.close()
    return samps2Check


# Define helper function to get distribution parameters
def get_distribution_parameters():
    return [(0.5, "CN1 Gaussian"), (1, "CN2 Gaussian"), (1.5, "CN3 gamma")]


######################################
# parsePlotExon
# Parses and plots exon-related data for a specific sample and exon.
#
# Args:
# - sampleName (str)
# - exonIndex (int)
# - exonOnSexChr (list): List indicating whether exons are on sex chromosomes.
# - samp2clusts (dict): Mapping of samples to clusters.
# - clust2samps (dict): Mapping of clusters to samples.
# - samples (list): List of sample names.
# - fitWith (dict): Mapping of clusters to control cluster ID.
# - exonsFPm (array): Exon-level FPM values.
# - exParams (array): Exon parameters.
# - paramsTitles (list): Titles of parameters.
# - exons (array): Exon information.
# - exp_loc (float): Exponential distribution location parameter.
# - exp_scale (float): Exponential distribution scale parameter.
# - matplotOpenFile: Opened PDF file for plotting.
def parsePlotExon(sampleName, exonIndex, exonOnSexChr, samp2clusts, clust2samps, samples,
                  fitWith, exonsFPM, exParams, paramsTitles, exons, exp_loc, exp_scale,
                  matplotOpenFile):

    # Get cluster ID based on exon's location
    if exonOnSexChr[exonIndex] == 0:
        clusterID = samp2clusts[sampleName][0]
    else:
        clusterID = samp2clusts[sampleName][1]

    # Get sample indexes for target and control clusters
    try:
        # Get sample indexes for the current cluster
        sampsInd = exonCalls.exonProcessing.getSampIndexes(clusterID, clust2samps, samples, fitWith)
    except Exception as e:
        logger.error("getClusterSamps failed for cluster %i : %s", clusterID, repr(e))
        raise

    # Extract FPM values for relevant samples
    exonFPM = exonsFPM[exonIndex, sampsInd]
    sampFPM = exonsFPM[exonIndex, samples.index(sampleName)]

    exonFilterState = exParams[clusterID][exonIndex, paramsTitles.index("filterStates")]
    logger.info(exonFilterState)
    ##### init graphic parameters
    # definition of a range of values that is sufficiently narrow to have
    # a clean representation of the adjusted distributions.
    xi = np.linspace(0, np.max(exonFPM), len(sampsInd) * 3)

    exonInfo = '_'.join(map(str, exons[exonIndex]))
    plotTitle = f"Cluster:{clusterID}, NbSamps:{len(sampsInd)}, exonInd:{exonIndex}\nexonInfos:{exonInfo}, filteringState:{exonFilterState}"

    yLists = []
    plotLegs = []
    verticalLines = [sampFPM]
    vertLinesLegs = f"{sampleName} FPM={sampFPM:.3f}"

    # get gauss_loc and gauss_scale from exParams
    gaussLoc = exParams[clusterID][exonIndex, paramsTitles.index("loc")]
    gaussScale = exParams[clusterID][exonIndex, paramsTitles.index("scale")]

    # if all the distributions can be represented,
    # otherwise a simple histogram and the fit of the exponential will be plotted.
    if exonFilterState > 1:
        # Update plot title with likelihood information
        plotTitle += f"\n{sampleName} likelihoods:\n"

        distribution_functions = CNVCalls.likelihoods.getDistributionObjects(exp_loc, exp_scale, gaussLoc, gaussScale)

        for ci in range(len(distribution_functions)):
            pdf_function, loc, scale, shape = distribution_functions[ci]
            # np.ndarray 1D float: set of pdfs for all samples
            # scipy execution speed up
            if shape is not None:
                # Si shape est défini (non None), utilisez-le dans l'appel de la fonction PDF
                PDFRanges = pdf_function(xi, loc=loc, scale=scale, a=shape)
                sampLikelihood = pdf_function(sampFPM, loc=loc, scale=scale, a=shape)
                plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}, shape={shape:.2f}]")
            else:
                # Si shape est None, n'incluez pas l'argument a dans l'appel de la fonction PDF
                PDFRanges = pdf_function(xi, loc=loc, scale=scale)
                sampLikelihood = pdf_function(sampFPM, loc=loc, scale=scale)
                plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}]")

            yLists.append(PDFRanges)
            plotTitle += f"CN{ci}:{sampLikelihood:.3e} "

        ylim = 2 * np.max(yLists[2])
    else:
        logger.info("the exon analysed is not covered, so no call")
        return
        
    figures.plots.plotExonProfile(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, matplotOpenFile)


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
    (countsFile, clustsFile, paramsFile, tocheckFile, pdfFile) = parseArgs(argv)

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
    # parse params clusters (parameters of continuous distributions fitted on CN0, CN2 coverage profil)
    try:
        (exParams, exp_loc, exp_scale, paramsTitles) = exonCalls.exonCallsFile.parseExonParamsFile(paramsFile)
    except Exception as e:
        raise Exception("parseParamsFile failed for %s : %s", paramsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing paramsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # parse user file
    try:
        samps2Check = parseUserTSV(tocheckFile, samples, exons)
    except Exception as e:
        raise Exception("parseUserTSV failed for %s : %s", tocheckFile, repr(e))

    #####################
    # process
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdfFile)
    # selects cluster-specific exons on autosomes, chrXZ or chrYW.
    exonOnSexChr = clusterSamps.genderPrediction.exonOnSexChr(exons)

    for sampleName in samps2Check.keys():
        for exonIndex in samps2Check[sampleName]:
            try:
                parsePlotExon(sampleName, exonIndex, exonOnSexChr, samp2clusts, clust2samps, samples,
                              fitWith, exonsFPM, exParams, paramsTitles, exons, exp_loc, exp_scale,
                              matplotOpenFile)
            except Exception as e:
                raise Exception("parsePlotExon failed for exon index %s sample %s : %s", str(exonIndex), sampleName, repr(e))

    matplotOpenFile.close()


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
