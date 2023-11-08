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
import callCNVs.exonProcessing
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
Given  three TSV files: one for fragment count data, another for sample cluster information,
and a user-defined file containing two columns [exonID, sampleID] tab-separated.
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
    --tocheck [str]: user-generated TSV file containing two columns [exonID(str), sampleID(str)] separated by tabs.
    --pdf[str]: a pdf file in which the graphical output will be produced.
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "tocheck=", "pdf="])
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
    return(countsFile, clustsFile, tocheckFile, pdfFile)


#########################
# parseUserTSV
# Parses a TSV file to map sample IDs to their respective exon indexes for autosomal and gonosomal exons.
#
# Args:
# - userTSV (str): Path to the user TSV file.
# - samples (list[str]): List of sample names.
# - autosomeExons (list of lists[str, int, int, str]): autosomal exon information [CHR, START, END, EXONID].
# - gonosomeExons (list of lists[str, int, int, str]): gonosomal exon information.
#
# Returns a tuple (samps2ExonInd_A, samps2ExonInd_G):
# - samps2ExonInd_A (dict): key==sampleID, value==List of exon indexes matching autosomal exons.
# - samps2ExonInd_G (dict): same as samps2ExonInd_A for gonosomes
def parseUserTSV(userTSV, samples, autosomeExons, gonosomeExons):
    # Helper function to find the index of an exon in a list of exons based on the exon identifier.
    def findExIndex(exonList, ei):
        return next((i for i, exon in enumerate(exonList) if exon[3] == ei), None)

    # Helper function to open a file, which can be a regular text file or a gzipped file.
    def openFile(filePath):
        return gzip.open(filePath, "rt") if filePath.endswith(".gz") else open(filePath)

    if not os.path.exists(userTSV):
        raise FileNotFoundError(f"User TSV file {userTSV} does not exist.")

    # Initialize dictionaries to hold the mapping of sample IDs to exon indices.
    samps2ExonInd_A, samps2ExonInd_G = {}, {}

    # Open the TSV file and iterate over each line.
    with openFile(userTSV) as file:
        for line in file:
            # Split each line into sample ID and exon ENST identifier.
            sID, exENST = line.rstrip().split("\t", maxsplit=1)
            # Check if the sample ID is in the list of valid samples.
            if sID in samples:
                # Find the exon index in autosomal and gonosomal lists.
                ei_A = findExIndex(autosomeExons, exENST)
                ei_G = findExIndex(gonosomeExons, exENST) if ei_A is None else None

                # Update the dictionaries with the found exon indices.
                if ei_A is not None:
                    samps2ExonInd_A.setdefault(sID, []).append(ei_A)
                elif ei_G is not None:
                    samps2ExonInd_G.setdefault(sID, []).append(ei_G)
                else:
                    logger.error(f"Exon identifier {exENST} not found in provided exon lists.")
                    raise ValueError(f"Exon identifier {exENST} not found.")
            else:
                logger.error(f"Sample name {sID} not found in provided samples list.")
                raise ValueError(f"Sample name {sID} not found.")

    return (samps2ExonInd_A, samps2ExonInd_G)


#############################################################
# filterExonsAndComputeCN2Params
# filter exons based on their FPM values and compute parameters for copy number variation (CNV) analysis.
# It evaluates each exon to determine its filtering state and calculates the mean and standard deviation
# for the FPMs of exons that are not filtered out.
# If an error occurs during the filtering or calculation, the function logs the error and raises an exception.
#
# Args:
# - targetFPMs (np.ndarray[floats]): FPM values for exons across cluster samples.
# - uncaptThreshold [float]: A threshold value for filtering out uncaptured FPMs.
#
# Returns a tuple (CN2params, filterStatesVec):
# - CN2ParamsArray (np.ndarray[floats]): calculated robustly Gaussian parameters for each exon,
#                                       including mean (loc) and standard deviation (scale)
#                                       for exons passing the two firsts filtering criteria
#                                       ("notCaptured", "cannotFitRG").
# - filterStatesVec (np.ndarray[ints]): filter state index of each exon.
def filterExonsAndComputeCN2Params(targetFPMs, uncaptThreshold):
    # Possible filtering states for exons
    filterStates = ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"]

    # Define parameter identifiers for CN2 metrics
    paramsID = ["loc", "scale"]

    # Initialize an array to store CN2 parameters with a default value of -1
    CN2ParamsArray = np.full((targetFPMs.shape[0], len(paramsID)), -1, dtype=np.float64)

    # Vector to keep track of the filter state for each exon
    filterStatesVec = np.zeros(targetFPMs.shape[0], dtype=int)

    # Process each exon to determine its filter state and compute CN2 parameters
    for exonIndex, exFPMs in enumerate(targetFPMs):
        try:
            # Apply filtering and update parameters for the current exon
            filterState, exonCN2Params = callCNVs.exonProcessing.evaluateAndComputeExonMetrics(exFPMs, uncaptThreshold, paramsID)
            filterStatesVec[exonIndex] = filterStates.index(filterState)
            if filterState == 'call':
                # Update CN2ParamsArray only for exon calls
                CN2ParamsArray[exonIndex] = exonCN2Params
        except Exception as e:
            logger.error("Error evaluateAndComputeExonMetrics exon index %i: %s", exonIndex, repr(e))
            raise

    return (CN2ParamsArray, filterStatesVec)


########################
# generateClusterExonPlots
# processes exon data for a given set of samples, identifying clusters of samples and generating
# plots for each exon within these clusters.
# It filters the exons based on predefined conditions, computes the copy number variation (CNV) parameters,
# and creates plots for exons that pass the filtering criteria.
# If plotting fails due to an exception, the function logs the error and proceeds with the next exon.
# The function ensures all plots are saved into a single PDF file.
#
# Args:
# - chromType [int]: type of chrom being processed (0 autosomes, 1 gonosomes).
# - samps2Check (dict): key==sampleID, value==exon indexes list.
# - FPMsArray (np.ndarray[floats]): FPM for a given exon across samples.
# - exonsInfos (list[str, int, int, str]): information about each exon[CHR, START, END, EXONID].
# - samples (list[strs]):sample IDs corresponding to the columns in FPMsArray.
# - clust2samps (dict): key==clusterID, value==sampleIDs list.
# - samp2clusts (dict): key==sampleID, value==their respective clusterIDs [autosomes, gonosomes].
# - fitWith (dict): key==clusterID, value== fitwith clusterIDs list.
# - hnorm_loc [float]: half Gaussian mean.
# - hnorm_scale: half Gaussian stdev.
# - unCaptFPMLimit: A threshold value for filtering out uncaptured FPMs in the analysis.
# - pdfFile: The file path for the PDF where the plots will be saved.
def generateClusterExonPlots(chromType, samps2Check, FPMsArray, exonsInfos, samples, clust2samps, samp2clusts,
                             fitWith, hnorm_loc, hnorm_scale, unCaptFPMLimit, pdfFile):
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdfFile)

    samp2Index = {sample: i for i, sample in enumerate(samples)}

    for sampID in samps2Check:
        clusterID = samp2clusts[sampID][chromType]
        exonIndexes = np.array(samps2Check[sampID])  # Assurez-vous que c'est un np.array d'entiers
        
        # Vérifiez que les indices d'exon sont dans les limites
        if np.any(exonIndexes >= FPMsArray.shape[0]):
            raise IndexError(f"Exon index out of bounds for cluster {clusterID}. Max allowed index is {FPMsArray.shape[0]-1}, but got index {exonIndexes.max()}.")

        sampsInClust = callCNVs.exonProcessing.getSampIndexes(clusterID, clust2samps, samp2Index, fitWith)
        sampsInClust = np.array(sampsInClust)  # Assurez-vous que c'est un np.array d'entiers
        
        # Vérifiez que les indices des échantillons sont dans les limites
        if np.any(sampsInClust >= FPMsArray.shape[1]):
            raise IndexError(f"Sample index out of bounds for cluster {clusterID}. Max allowed index is {FPMsArray.shape[1]-1}, but got index {sampsInClust.max()}.")

        # Si aucune erreur, procéder avec l'accès sécurisé
        targetFPMs = FPMsArray[np.ix_(exonIndexes, sampsInClust)]

        CN2params, filterStatesVec = filterExonsAndComputeCN2Params(targetFPMs, unCaptFPMLimit)

        for ei in exonIndexes:
            sampExFPM = FPMsArray[ei, samp2Index[sampID]]  # Direct lookup from the dictionary
            if filterStatesVec[ei] < 2:
                warning_message = f"Skipping plot for exon index {ei}, sample {sampID}, cluster {clusterID}..."
                logger.warning(warning_message)
                continue

            try:
                getReadyPlotExon(sampID, ei, clusterID, sampExFPM, targetFPMs[ei], CN2params[ei], filterStatesVec[ei],
                                 exonsInfos[ei], hnorm_loc, hnorm_scale, matplotOpenFile)
            except Exception as e:
                error_message = f"Plotting failed for exon index {ei}, sample {sampID}, cluster {clusterID}: {e}"
                logger.error(error_message)
                continue  # Continue with the next exon

    matplotOpenFile.close()


######################################
# getReadyPlotExon
# Parses and plots exon-related data for a specific sample and exon.
#
# Args:
# - sampID [str]
# - ei [int]
# - clusterID [int]
# - sampExFPM [float]: sample exon FPM
# - exClustFPMs (np.ndarray[floats]): exon FPM values for all samples in clusterID
# - exCN2params (np.ndarray[floats]): exon gaussian parameters [loc, scale]
# - exfilterState [int]: filter index
# - exInfos (list[str, int, int, str]): exon [CHR, START, END, EXONID]
# - hnorm_loc (float): distribution location parameter.
# - hnorm_scale (float): distribution scale parameter.
# - matplotOpenFile: Opened PDF file for plotting.
def getReadyPlotExon(sampID, ei, clusterID, sampExFPM, exClustFPMs, exCN2params, exfilterState, exInfos,
                     hnorm_loc, hnorm_scale, matplotOpenFile):
    ##### init graphic parameters
    # definition of a range of values that is sufficiently narrow to have
    # a clean representation of the adjusted distributions.
    xi = np.linspace(0, np.max(sampExFPM), len(exClustFPMs) * 3)

    exonInfo = '_'.join(map(str, exInfos))
    plotTitle = f"Cluster:{clusterID}, NbSamps:{len(exClustFPMs)}, exonInd:{ei}\nexonInfos:{exonInfo}, filteringState:{exfilterState}"

    yLists = []
    plotLegs = []
    verticalLines = [sampExFPM]
    vertLinesLegs = f"{sampID} FPM={sampExFPM:.3f}"

    # get gauss_loc and gauss_scale from exParams
    gaussLoc = exCN2params[0]
    gaussScale = exCN2params[1]

    # Update plot title with likelihood information
    plotTitle += f"\n{sampID} likelihoods:\n"

    distribution_functions = CNVCalls.likelihoods.getDistributionObjects(hnorm_loc, hnorm_scale, gaussLoc, gaussScale)

    for ci in range(len(distribution_functions)):
        pdf_function, loc, scale, shape = distribution_functions[ci]
        # np.ndarray 1D float: set of pdfs for all samples
        # scipy execution speed up
        if shape is not None:
            PDFRanges = pdf_function(xi, loc=loc, scale=scale, a=shape)
            sampLikelihood = pdf_function(sampExFPM, loc=loc, scale=scale, a=shape)
            plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}, shape={shape:.2f}]")
        else:
            PDFRanges = pdf_function(xi, loc=loc, scale=scale)
            sampLikelihood = pdf_function(sampExFPM, loc=loc, scale=scale)
            plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}]")

        yLists.append(PDFRanges)
        plotTitle += f"CN{ci}:{sampLikelihood:.3e} "

    ylim = 2 * np.max(yLists[2])

    plotExonProfile(exClustFPMs, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, matplotOpenFile)


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
    (countsFile, clustsFile, tocheckFile, pdfFile) = parseArgs(argv)

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
    # parse user file
    try:
        (samps2ExonInd_A, samps2ExonInd_G) = parseUserTSV(tocheckFile, samples, autosomeExons, gonosomeExons)
    except Exception as e:
        raise Exception("parseUserTSV failed for %s : %s", tocheckFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing UserTSV, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ###################
    # fit CN0
    try:
        (hnorm_loc, hnorm_scale, uncaptThreshold) = callCNVs.exonProcessing.computeCN0Params(intergenicFPMs)
    except Exception as e:
        raise Exception("computeCN0Params failed: %s", repr(e))

    thisTime = time.time()
    logger.debug("Done computeCN0Params, loc=%.2f, scale=%.2f, uncapThreshold=%.2f in %.2f s",
                 hnorm_loc, hnorm_scale, uncaptThreshold, thisTime - startTime)
    startTime = thisTime

    # build root name for output pdfs, will just need to append autosomes.pdf or gonosomes.pdf
    rootFileName = os.path.basename(pdfFile)
    rootFileName = os.path.splitext(rootFileName)[0]

    # autosomes
    try:
        chromType = 0
        pdfFile_A = rootFileName + "_autosomes.pdf"
        generateClusterExonPlots(chromType, samps2ExonInd_A, autosomeFPMs, autosomeExons, samples, clust2samps, samp2clusts,
                                 fitWith, hnorm_loc, hnorm_scale, uncaptThreshold, pdfFile_A)
    except Exception as e:
        logger.error("generateClusterExonPlots failed for autosomes: %s", repr(e))
        raise

    # sex chromosomes
    try:
        chromType = 1
        pdfFile_G = rootFileName + "_gonosomes.pdf"
        generateClusterExonPlots(chromType, samps2ExonInd_G, gonosomeFPMs, gonosomeExons, samples, clust2samps, samp2clusts,
                                 fitWith, hnorm_loc, hnorm_scale, uncaptThreshold, pdfFile_G)
    except Exception as e:
        logger.error("generateClusterExonPlots failed for autosomes: %s", repr(e))
        raise

    thisTime = time.time()
    logger.debug("Done exon profil plots, in %.2f s", thisTime - startTime)
    startTime = thisTime


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
