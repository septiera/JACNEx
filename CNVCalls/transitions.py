import logging
import numpy as np
import os
import matplotlib.pyplot
import matplotlib.backends.backend_pdf

import figures.plots
import clusterSamps.smoothing


# prevent numba flooding the logs when we are in DEBUG loglevel
logging.getLogger('numba').setLevel(logging.WARNING)
# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
# getTransMatrix
# Given likelihoods and priors, calculates the copy number transition matrix.
# Filters out irrelevant samples, determines most probable states for exons,
# and normalizes transitions.
# Includes a 'void' state incorporating priors for better initialization and restart
# in the personalised Viterbi algorithm.
#
#
# Also generates a bar plot of copy number counts by samples (only if logger debug mode).
#
# Args:
# - likelihoods(dict): keys = sampID[str], values = np.ndarray of likelihoods[floats]
#                         dim = nbOfExons * NbOfCNStates
# - chr2Exons (dict): keys = chrID[str], values = startExonIndex[int]
# - priors (np.ndarray[floats]): prior probabilities for each copy number status.
# - CNStates (list[strs]): Names of copy number types.
# - samp2clusts (dict): keys = sampID[str], values = [clustID_A, clustID_G][strs]
# - fitWith (dict): keys = clusterID[str], values = clusterIDs list  
# - plotDir[str] : directory to save the graphical representation.
#
# Returns:
# - transMatVoid (np.ndarray[floats]): transition matrix used for the hidden Markov model,
#                                      including the "void" state.
#                                      dim = [nbStates+1, nbStates+1]
def getTransMatrix(likelihoods_A, likelihoods_G, chr2Exons_A, chr2Exons_G,
                   priors, CNStates, samp2clusts, fitWith, plotDir):
    nbStates = len(CNStates)
    # 2D array, expected format for a transition matrix [i; j]
    # contains all prediction counts of states, taking into account
    # the preceding states for the entire sampling.
    transitions = np.zeros((nbStates, nbStates), dtype=int)

    try:
        # Get counts for CN levels of autosomes and update the transition matrix
        transitions, CNcounts_A, samp2CNEx_A = countsCNStates(likelihoods_A, nbStates, transitions,
                                                              priors, chr2Exons_A)
        # Get counts for CN levels of gonosomes and update the transition matrix
        transitions, CNcounts_G, samp2CNEx_G = countsCNStates(likelihoods_G, nbStates, transitions,
                                                              priors, chr2Exons_G)
    except Exception as e:
        logger.error(repr(e))
        raise

    ### Format transition matrix
    # normalize each row to ensure sum equals 1
    # not require normalization with the total number of samples
    row_sums = np.sum(transitions, axis=1, keepdims=True)
    normalized_arr = transitions / row_sums

    # add void status and incorporate priors for all current observations
    transMatVoid = np.vstack((priors, normalized_arr))
    transMatVoid = np.hstack((np.zeros((nbStates + 1, 1)), transMatVoid))

    #####################################
    ####### DEBUG PART ##################
    #####################################
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        try:
            debug_process(transMatVoid, CNcounts_A, samp2CNEx_A, CNcounts_G, samp2CNEx_G,
                          samp2clusts, fitWith, CNStates, plotDir)
        except Exception as e:
            logger.error("DEBUG follow process failed: %s", repr(e))
            raise

    return transMatVoid


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
# countsCNStates
# Counts the Copy Number (CN) states for exons based on likelihood data and
# update transition matrix.

# Args:
# - likelihoodDict (dict): keys=sampID, values=np.ndarray(NBExons,NBStates)
# - nbStates [int]
# - transitions (np.ndarray[floats]): A transition matrix to be updated.
#                                     dim = NBStates * NBStates
# - priors (np.ndarray): Prior probabilities for CN states.
# - exonOnChr (dict): A dictionary mapping chromosome IDs to exon indexes.

# Returns:
# - transitions (np.ndarray): An updated transition matrix.
# - samp2CNCounts (dict): event counts per sample, keys = sampID,
#   values = numpy arrays 1D [CN0counts, CN1counts, CN2counts, CN3counts]
# - samp2CNEx (dict): CN states list per sample, keys = sampID,
#   values = 1D numpy arrays 1D (NbExons) representing CN states (0 to 3)
#   and no call (-1)
def countsCNStates(likelihoodDict, nbStates, transitions, priors, exonOnChr):
    # Initialize dictionaries
    samp2CNCounts = {}
    samp2CNEx = {}
    
    # Get the exon indexes start positions of chromosomes as a set for faster look-up
    chrStarts = list(exonOnChr.values())

    for sampID, likelihoods in likelihoodDict.items():
        prevCN = np.argmax(priors)  # Normal CN status index = 2
        CNsVec = np.full(likelihoods.shape[0], -1, dtype=int)
        countsVec = np.zeros(nbStates, dtype=int)

        # Calculate the most probable CN state based on probabilities and priors
        odds = likelihoods * priors
        CNsList = np.argmax(odds, axis=1)

        # Check for no interpretable data
        skip_exon = np.any(likelihoods == -1, axis=1)

        for ei, is_skipped in enumerate(skip_exon):
            currCN = CNsList[ei]
            if is_skipped:
                continue

            if ei in chrStarts:
                prevCN = np.argmax(priors)

            transitions[prevCN, currCN] += 1
            CNsVec[ei] = currCN
            prevCN = currCN
            countsVec[currCN] += 1

        samp2CNCounts[sampID] = countsVec
        samp2CNEx[sampID] = CNsVec

    return (transitions, samp2CNCounts, samp2CNEx)


############################################
# debug_process
# Debugging and data processing function for CN (Copy Number) analysis.
# 1. Logs the transition matrix with formatting.
# 2. Retrieves unique sample IDs from dictionaries.
# 3. Logs a summary of CN levels for exons in autosomes and gonosomes.
# 4. Prepares data for plotting.
# 5. Creates a PDF file for curve plots.
# 6. Iterates through clusters, processes data, and generates cluster-specific plots.
# 7. Closes the PDF file.
#
# Args:
# - transMatVoid (list of lists[floats]): transition matrix used for the hidden Markov model,
#                                         including the "void" state.
#                                         dim = [nbStates+1, nbStates+1]
# - samp2CNcounts_A (dict): CN state counts for autosomes.
#                           keys=sampID, values=np.ndarray[NBCN0,NBCN1,NBCN2,NBCN3]
# - samp2CNcounts_G (dict): CN state counts for gonosomes.
# - samp2clusts (dict): Sample-to-cluster mapping.
# - CNStates (list[strs]): List of CN states.
# - plotFile (str): File name for saving the plot.
#
# Returns:
# - None
def debug_process(transMatVoid, CNcounts_A, samp2CNEx_A, CNcounts_G, samp2CNEx_G,
                  samp2clusts, fitWith, CNStates, plotDir):
    ##############
    # Log the transition matrix with formatting
    logger.debug("#### Transition Matrix (Formatted) #####")
    for row in transMatVoid:
        row2Print = ' '.join(format(num, ".3e") for num in row)
        logger.debug(row2Print)

    ##############
    # Log the counting summary
    logger.debug("#### Counting Summary of CN Levels for Exons in Autosomes and Gonosomes #####")

    # Get all unique sample IDs from both dictionaries
    sampIDsList = list(set(samp2CNEx_A.keys()) | set(samp2CNEx_G.keys()))

    # Initialize lists for plotting and csluster data
    countsList2Plot = []
    clust2counts = {}

    testFile = os.path.join(plotDir, "Event_countsBySamples.tsv")
    try:
        with open(testFile, "x") as outFH:
            header = "\t".join(["SAMPID", "clustID_A", "CN0_A", "CN1_A", "CN2_A", "CN3_A",
                                "clustID_G", "CN0_G", "CN1_G", "CN2_G", "CN3_G"])
            outFH.write(header + "\n")

            for sampID in sampIDsList:
                CN_A = CNcounts_A.get(sampID, np.zeros(4, dtype=int))
                CN_G = CNcounts_G.get(sampID, np.zeros(4, dtype=int))

                log_elements = [
                    sampID,
                    samp2clusts[sampID][0],
                    '\t'.join(map(str, CN_A)),
                    samp2clusts[sampID][1],
                    '\t'.join(map(str, CN_G))]

                row = '\t'.join(map(str, log_elements))
                outFH.write(row + "\n")

                countsList2Plot.append(np.concatenate((CN_A, CN_G)).tolist())
                if samp2clusts[sampID][0] not in clust2counts:
                    clust2counts[samp2clusts[sampID][0]] = []
                clust2counts[samp2clusts[sampID][0]].append(CN_A)
                if samp2clusts[sampID][1] not in clust2counts:
                    clust2counts[samp2clusts[sampID][1]] = []
                clust2counts[samp2clusts[sampID][1]].append(CN_G)

    except Exception as e:
        logger.error("Cannot open testFile %s: %s", testFile, e)
        raise Exception('Cannot open outFile')

    ###############################
    # print dans un fichier de debug des évenements spécifiques a un type de CN
    # que pour CN0 autosomes pour l'instant
    logger.debug("#### Evaluation of events and their recurrence #####")

    clusterToProcess = "A_01"

    for CNtype in range(len(CNStates)):
        if CNtype == 2:
            continue

        clustEventInd = {}

        for sampID, clust in samp2clusts.items():
            if clust[0] != clusterToProcess:
                continue

            CNsPath = samp2CNEx_A.get(sampID)
            if CNsPath is not None:
                eventInd = np.where(CNsPath == CNtype)[0]
                for event in eventInd:
                    clustEventInd[event] = clustEventInd.get(event, np.zeros(len(sampIDsList), dtype=int))
                    clustEventInd[event][sampIDsList.index(sampID)] = 1

        testFile = os.path.join(plotDir, f"Event_CN{CNtype}.tsv")
        try:
            with open(testFile, "x") as outFH:
                header = "exonIndex\t" + "\t".join(sampIDsList) + "\n"
                outFH.write(header)

                for i, counts in clustEventInd.items():
                    row = f"{i}\t" + '\t'.join(map(str, counts)) + "\n"
                    outFH.write(row)
        except Exception as e:
            logger.error(f"Cannot open testFile {testFile}: {e}")
            raise Exception('cannot open outFile')

    ##########################
    #
    logger.debug("#### Plotting data... #####")
    # Plot counts distribution barplot
    # represents the frequencies of events for the entire cohort (CN0_A, CN1_A,CN2_A,CN3_A, CN0_G, CN1_G,CN2_G,CN3_G)
    try:
        plotFile = os.path.join(plotDir, "CN_Frequencies_Likelihoods_Plot.pdf")
        figures.plots.barPlot(countsList2Plot, [term + "_A" for term in CNStates] + [term + "_G" for term in CNStates], plotFile)
    except Exception as e:
        logger.error("barPlot failed: %s", repr(e))
        raise

    # plot for each cluster of event count profiles (by CNstates)
    curvePlotFile = os.path.join(plotDir, "CN_Exons_Plot.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(curvePlotFile)

    try:
        process_cluster(clust2counts, fitWith, CNStates, pdf)
    except Exception as e:
        logger.debug(e)

    pdf.close()


############################################
# process_cluster
# This function processes and plots count data for a specific cluster, including:
# - Plotting smoothed count curves.
# - Showing mean and median lines.
# - Configuring plot labels and legends.
#
# Args:
# - clust2Counts (dict):
# - fitWith (dict)
# - CNStates (list[strs]):
# - pdf (matplotlib.pyplot)
def process_cluster(clust2Counts, fitWith, CNStates, pdf):
    # Loop through the keys (clusterID) of the dictionary in alphabetical order
    for cluster_id in sorted(clust2Counts.keys()):
        counts_list = np.array(clust2Counts[cluster_id])

        for CNstate in CNStates:
            CNcountsCurrent = counts_list[:, CNStates.index(CNstate)]

            # logger.debug("cluster %s, CNstate %s shape  %i", cluster_id, CNstate, len(CNcounts))
            if np.all(CNcountsCurrent == 0):
                logger.debug("%s invalid cluster contains no CN counts", cluster_id)
                continue

            # Calculate statistics
            mean = np.mean(CNcountsCurrent)
            median = np.median(CNcountsCurrent)
            stdev = np.std(CNcountsCurrent)

            # Create a new figure for each cluster and CNState
            fig, ax = matplotlib.pyplot.subplots()

            try:
                plot_histogram_and_density(ax, CNcountsCurrent, f'{cluster_id} counts', 'skyblue')
            except Exception as e:
                logger.error("%s", repr(e))

            # Plot mean and median lines
            ax.axvline(mean, color='grey', label=f'mean={mean:.3f}')
            ax.axvline(median, color='green', label=f'median={median:.3f}')

            # Plot standard deviation lines (conditional)
            if CNStates.index(CNstate) != 2:
                ax.axvline((mean + stdev), linestyle='--', color='grey', label=f'stdev=mean+{stdev:.3f}')
            else:
                ax.axvline((mean - stdev), linestyle='--', color='grey', label=f'stdev=mean-{stdev:.3f}')

            # counts_listFitWith = None
            # if fitWith[cluster_id]:
            #     for fitWithID in fitWith[cluster_id]:
            #         counts_listFitWith = np.array(clust2Counts[fitWithID])
            #         CNcountsFitWith = counts_listFitWith[:, CNStates.index(CNstate)]
            #         meanFitWith = np.mean(CNcountsFitWith)
            #         stdevFitWith = np.std(CNcountsFitWith)
            #         plot_histogram_and_density(ax, CNcountsFitWith, f"{fitWithID} counts", 'red')
            #         ax.axvline(meanFitWith, color='red', label=f'mean={meanFitWith:.3f}, stdev={stdevFitWith:.3f}')

            if CNStates.index(CNstate) == 2:
                ax.set_xlim(max(CNcountsCurrent) / 1.1, max(CNcountsCurrent) * 1.1)

            # Set labels and title
            ax.set_xlabel('CN exon counts')
            ax.set_ylabel('Densities')
            ax.set_title(f'Cluster {cluster_id} - NBsamps {len(CNcountsCurrent)} - CNState {CNstate}')
            ax.legend(loc='upper right', fontsize='small')

            pdf.savefig()
            matplotlib.pyplot.close(fig)


def plot_histogram_and_density(ax, data, label, color):
    # Calculate the interquartile range (IQR)
    # iqr = np.percentile(CNcountsCurrent, 75) - np.percentile(CNcountsCurrent, 25)
    # Freedman-Diaconis
    # bins = int(2.0 * iqr * len(CNcountsCurrent) ** (-1/3))
    # bins = int(1 + np.sqrt(len(CNcountsCurrent)))
    # bins = int(1 + np.log2(len(data)))
    bins = len(data) // 2
    ax.hist(data, bins=bins, color=color, alpha=0.5, label=label, density=True)
    # Smooth and plot count curves
    # testbw = ['ISJ', 'scott', 'silverman']
    # colors = ["gray", "red", "orange", "green", "blue", "purple"]
    # for bwInd in range(len(testbw)):
    # dr, dens, bwValue = clusterSamps.smoothing.smoothData(data, maxData=max(data), numValues=max(data) * 100, bandwidth='ISJ')
    # ax.plot(dr, dens, color=color, label=f'bw={bwValue:.3f}')
