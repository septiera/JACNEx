###############################################################################################
##################################### test_exonViewer #########################################
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
import matplotlib.pyplot
import matplotlib.backends.backend_pdf

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import exonCalls.exonCallsFile
import exonCalls.exonProcessing
import CNVCalls.likelihoods

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
specific distributions will be fitted for each copy number (CN0: half normal, CN1: Gaussian,
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
                    followed by distribution parameters ["loc", "scale"] for half normal
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
#
# Args:
# - userTSV (str): Path to the user TSV file.
# - samples (list[str]): List of sample names.
# - exons (list of lists[str, int, int, str]): List of exon definitions [CHR, START, END, EXONID].
# - autosomeExons (list of lists[str, int, int, str]): autosomal exon information [CHR, START, END, EXONID].
# - gonosomeExons (list of lists[str, int, int, str]): gonosomal exon information.
#
# Returns:
# - samps2Check (dict): keys = sampleID, and values are sub-dictionaries categorized as follows:
#                       - "A": List of exon indexes matching autosomal exons.
#                       - "G": List of exon indexes matching gonosomal exons.
def parseUserTSV(userTSV, samples, autosomeExons, gonosomeExons):
    try:
        # Check if the file exists before opening it
        if not os.path.exists(userTSV):
            raise FileNotFoundError("User TSV file does not exist.")

        # Open the file using 'with' to ensure proper file closure
        with (gzip.open(userTSV, "rt") if userTSV.endswith(".gz") else open(userTSV, "r")) as userTSVFH:
            samps2Check = {key: {"A": [], "G": []} for key in samples}

            for line in userTSVFH:
                splitLine = line.rstrip().split("\t", maxsplit=1)
                sampID = splitLine[0]
                exonENST = splitLine[1]

                if sampID in samples:
                    exonIDFound = False
                    exonID_A = 0
                    exonID_G = 0

                    while not exonIDFound and exonID_A < len(autosomeExons):
                        exonType = "A"  # Assume it's an autosome exon by default
                        if autosomeExons[exonID_A][3] == exonENST:
                            samps2Check[sampID][exonType].append(exonID_A)
                            exonIDFound = True
                        else:
                            exonID_A += 1

                    # If no match is found in autosomes and there are gonosome exons
                    if not exonIDFound and gonosomeExons:
                        while not exonIDFound and exonID_G < len(gonosomeExons):
                            exonType = "G"  # Assume it's a gonosome exon
                            if gonosomeExons[exonID_G][3] == exonENST:
                                samps2Check[sampID][exonType].append(exonID_G)
                                exonIDFound = True
                            else:
                                exonID_G += 1

                    if not exonIDFound:
                        logger.error("Exon identifier %s has no match with the bed file provided in step 1.", exonENST)
                        raise Exception("Exon identifier not found.")
                else:
                    logger.error("Sample name %s has no match with the sample name list.", sampID)
                    raise Exception("Sample name not found.")

    except Exception as e:
        # Handle exceptions and log errors
        logger.error("Error processing userTSV: %s", e)
        raise e

    return samps2Check


######################################
# parsePlotExon
# Parses and plots exon-related data for a specific sample and exon.
#
# Args:
# - sampID (str)
# - exonType (str): "A" or "G"
# - ei (int)
# - samp2clusts (dict): Mapping of samples to clusters.
# - clust2samps (dict): Mapping of clusters to samples.
# - samples (list): List of sample names.
# - fitWith (dict): Mapping of clusters to control cluster ID.
# - fpmData (array): Exon-level FPM values.
# - exonMetrics (array): Exon parameters.
# - metricsNames (list): Titles of parameters.
# - exonsData (array): Exon information.
# - hnorm_loc (float): distribution location parameter.
# - hnorm_scale (float): distribution scale parameter.
# - matplotOpenFile: Opened PDF file for plotting.
def parsePlotExon(sampID, exonType, ei, samp2clusts, clust2samps, samples, fitWith, fpmData, exonMetrics,
                  metricsNames, exonsData, hnorm_loc, hnorm_scale, matplotOpenFile):

    # Get cluster ID based on exon's location
    if exonType == "A":
        clusterID = samp2clusts[sampID][0]
    else:
        clusterID = samp2clusts[sampID][1]

    # Get sample indexes for target and control clusters
    try:
        # Get sample indexes for the current cluster
        sampsInd = exonCalls.exonProcessing.getSampIndexes(clusterID, clust2samps, samples, fitWith)
    except Exception as e:
        logger.error("getClusterSamps failed for cluster %i : %s", clusterID, repr(e))
        raise

    # Extract FPM values for relevant samples
    exonFPM = fpmData[ei, sampsInd]
    sampFPM = fpmData[ei, samples.index(sampID)]

    exonFilterState = exonMetrics[clusterID][ei, metricsNames.index("filterStates")]
    logger.info(exonFilterState)
    ##### init graphic parameters
    # definition of a range of values that is sufficiently narrow to have
    # a clean representation of the adjusted distributions.
    xi = np.linspace(0, np.max(exonFPM), len(sampsInd) * 3)

    exonInfo = '_'.join(map(str, exonsData[ei]))
    plotTitle = f"Cluster:{clusterID}, NbSamps:{len(sampsInd)}, exonInd:{ei}\nexonInfos:{exonInfo}, filteringState:{exonFilterState}"

    yLists = []
    plotLegs = []
    verticalLines = [sampFPM]
    vertLinesLegs = f"{sampID} FPM={sampFPM:.3f}"

    # get gauss_loc and gauss_scale from exParams
    gaussLoc = exonMetrics[clusterID][ei, metricsNames.index("loc")]
    gaussScale = exonMetrics[clusterID][ei, metricsNames.index("scale")]

    # if all the distributions can be represented,
    # otherwise a simple histogram and the fit of the half normal will be plotted.
    if exonFilterState > 1:
        # Update plot title with likelihood information
        plotTitle += f"\n{sampID} likelihoods:\n"

        distribution_functions = CNVCalls.likelihoods.getDistributionObjects(hnorm_loc, hnorm_scale, gaussLoc, gaussScale)

        for ci in range(len(distribution_functions)):
            pdf_function, loc, scale, shape = distribution_functions[ci]
            # np.ndarray 1D float: set of pdfs for all samples
            # scipy execution speed up
            if shape is not None:
                PDFRanges = pdf_function(xi, loc=loc, scale=scale, a=shape)
                sampLikelihood = pdf_function(sampFPM, loc=loc, scale=scale, a=shape)
                plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}, shape={shape:.2f}]")
            else:
                PDFRanges = pdf_function(xi, loc=loc, scale=scale)
                sampLikelihood = pdf_function(sampFPM, loc=loc, scale=scale)
                plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}]")

            yLists.append(PDFRanges)
            plotTitle += f"CN{ci}:{sampLikelihood:.3e} "

        ylim = 2 * np.max(yLists[2])
    else:
        logger.info("The analyzed exon is not covered, so no call is made.")
        return

    plotExonProfile(exonFPM, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, matplotOpenFile)


#########################
# plotExonProfile
# plots a density histogram for a raw data list, along with density or distribution
# curves for a specified number of data lists. It can also plot vertical lines to mark
# points of interest on the histogram. The graph is saved as a PDF file.
#
# Args:
# - rawData (np.ndarray[float]): exon FPM counts
# - xi (list[float]): x-axis values for the density or distribution curves, ranges
# - yLists (list of lists[float]): y-axis values, probability density function values
# - plotLegs (list[str]): labels for the density or distribution curves
# - verticalLines (list[float]): vertical lines to be plotted, FPM tresholds
# - vertLinesLegs (list[str]): labels for the vertical lines to be plotted
# - plotTitle [str]: title of the plot
# - pdf (matplotlib.backends object): a file object for save the plot
def plotExonProfile(rawData, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, pdf):
    # Disable interactive mode to prevent display of the plot during execution
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(8, 8))
    # Plot a density histogram of the raw data with a number of bins equal to half the number of data points
    matplotlib.pyplot.hist(rawData, bins=int(len(rawData) / 2), density=True)

    # Plot the density/distribution curves for each set of x- and y-values
    if len(yLists) > 1:
        for i in range(len(yLists)):
            matplotlib.pyplot.plot(xi, yLists[i], label=plotLegs[i])

    # Plot vertical lines to mark points of interest on the histogram
    if verticalLines:
        matplotlib.pyplot.axvline(verticalLines, color="red", linestyle='dashdot', linewidth=1, label=vertLinesLegs)

    # Set the x- and y-axis labels, y-axis limits, title, and legend
    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("probability density (=likelihoods)")
    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.title(plotTitle)
    matplotlib.pyplot.legend(loc='upper right', fontsize='small')

    pdf.savefig(fig)
    matplotlib.pyplot.close()


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
    # parse and FPM-normalize the counts, distinguishing between exons and intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("Failed to parse and normalize counts for %s: %s", countsFile, repr(e))
        raise Exception("Failed to parse and normalize counts")

    thisTime = time.time()
    logger.debug("Done parsing and normalizing counts file, in %.2f s", thisTime - startTime)
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
        (exonMetrics_A, exonMetrics_G, hnorm_loc, hnorm_scale, metricsNames) = exonCalls.exonCallsFile.parseExonParamsFile(paramsFile)
    except Exception as e:
        raise Exception("parseParamsFile failed for %s : %s", paramsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing paramsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # parse user file
    try:
        samps2Check = parseUserTSV(tocheckFile, samples, autosomeExons, gonosomeExons)
    except Exception as e:
        raise Exception("parseUserTSV failed for %s : %s", tocheckFile, repr(e))

    #####################
    # process
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdfFile)

    for sampID in samps2Check.keys():
        for exonType in ["A", "G"]:
            exonIndexes = samps2Check[sampID][exonType]
            if exonIndexes:  # Check if the list is not empty
                exonMetrics = exonMetrics_A if exonType == "A" else exonMetrics_G
                fpmData = autosomeFPMs if exonType == "A" else gonosomeFPMs
                exonsData = autosomeExons if exonType == "A" else gonosomeExons

                for ei in exonIndexes:
                    try:
                        parsePlotExon(sampID, exonType, ei, samp2clusts, clust2samps,
                                      samples, fitWith, fpmData, exonMetrics,
                                      metricsNames, exonsData, hnorm_loc, hnorm_scale,
                                      matplotOpenFile)
                    except Exception as e:
                        if exonType == "A":
                            error_message = f"Failed for exon index {ei} sample {sampID} from autosomes: {repr(e)}"
                        else:
                            error_message = f"Failed for exon index {ei} sample {sampID} from gonosomes: {repr(e)}"
                        raise Exception(error_message)

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
