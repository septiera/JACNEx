import logging
import numpy as np
import os
import matplotlib.pyplot
import matplotlib.backends.backend_pdf

import figures.plots

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
    #####
    # dev control options
    clusters2Process = ["A_01", "A_02", "A_03"]

    ##############
    logger.debug("#### Transition Matrix (Formatted) #####")
    # enables observation of the transition matrix to monitor probabilities
    for row in transMatVoid:
        row2Print = ' '.join(format(num, ".3e") for num in row)
        logger.debug(row2Print)

    ##############
    logger.debug("#### Save counting CNs/exons in Autosomes and Gonosomes #####")
    # initial CN predictions by exons, counting facilitating comparisons between samples and clusters
    try:
        countsList2Plot, clust2counts = saveCNCounts(samp2CNEx_A, samp2CNEx_G, CNcounts_A,
                                                     CNcounts_G, samp2clusts, plotDir, CNStates)
    except Exception as e:
        logger.error("Printing the event count summary failed due to: %s", repr(e))

    ##############
    logger.debug("#### Saving events by CN type excluding CN2 for the clusters to monitor #####")
    # Extraction of events by target cluster, enabling fine downstream control with tools like IGV
    try:
        saveEventByCluster(clusters2Process, samp2clusts, CNStates, samp2CNEx_A, plotDir)
    except Exception as e:
        logger.error("Printing events by CN type failed due to: %s", repr(e))

    ##############
    logger.debug("#### Barplot showing event count distribution for autosomes and gonosomes #####")
    try:
        plotFile = os.path.join(plotDir, "CN_Frequencies_Likelihoods_Plot.pdf")
        figures.plots.barPlot(countsList2Plot, [term + "_A" for term in CNStates] + [term + "_G" for term in CNStates], plotFile)
    except Exception as e:
        logger.error("barPlot failed: %s", repr(e))
        raise

    ##############
    logger.debug("#### Histogram plot per cluster representing counts of a specific CN type #####")
    try:
        curvePlotFile = os.path.join(plotDir, "CN_Exons_Plot.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(curvePlotFile)
        processCluster(clust2counts, fitWith, CNStates, pdf)
        pdf.close()
    except Exception as e:
        logger.debug("Histogram failed due to: %s", repr(e))


################################
# saveCNCounts
# Prints counting summary data to an output file and collects data for plotting
# and cluster-specific counts.
#
# Args:
# - CNcounts_A (dict): autosomal CN state counts(values list[NBofCNStates])
#                      for samples (keys).
# - CNcounts_G (dict): gonosomal CN state counts for samples.
# - samp2clusts (dict): mapping sample IDs to clusters.
# - plotDir (str): Directory path for storing the output file.
# - CNStates (list): List of CN states.
#
# Returns:
# tuple: A tuple containing two elements:
#     - countsList2Plot (list of list): CN state counts[ints] for plotting.
#                                       1 list = NBofCNStates_autosomes + NBofCNStates_gonosomes
#     - clust2counts (dict): keys = clusterID[str], values = list of lists[ints], each one containing
#                            CN type counts for a cluster sample
def saveCNCounts(CNcounts_A, CNcounts_G, samp2clusts, plotDir, CNStates):
    # Initialize variables
    countsList2Plot, clust2counts = [], {}

    # Define the path for the output file
    testFile = os.path.join(plotDir, "Event_countsBySamples.tsv")

    # Open the output file for writing
    with open(testFile, "w") as outFH:
        # Define the header for the output file
        header = "\t".join(["SAMPID", "clustID_A", "CN0_A", "CN1_A", "CN2_A", "CN3_A",
                            "clustID_G", "CN0_G", "CN1_G", "CN2_G", "CN3_G"])
        outFH.write(header + "\n")

        # Iterate through sample IDs and their associated clusters
        for sampID, (clust_A, clust_G) in samp2clusts.items():
            # Retrieve CN counts for autosomes and gonosomes, defaulting to zeros if not found
            CN_A = CNcounts_A.get(sampID, np.zeros(len(CNStates), dtype=int))
            CN_G = CNcounts_G.get(sampID, np.zeros(len(CNStates), dtype=int))

            # Create a list to represent the row to print
            toPrint = [sampID, clust_A] + list(CN_A) + [clust_G] + list(CN_G)

            # Convert the row to a tab-separated string and write it to the output file
            row2Print = '\t'.join(map(str, toPrint))
            outFH.write(row2Print + "\n")

            # Store the CN counts in the cluster-specific data dictionaries
            countsList2Plot.append(list(CN_A) + list(CN_G))
            clust2counts.setdefault(clust_A, []).append(list(CN_A))
            clust2counts.setdefault(clust_G, []).append(list(CN_G))

    return countsList2Plot, clust2counts


##################################
# saveEventByCluster
# Organizes and stores event data based on clusters and CN types in separate output files.
# It goes through a list of cluster IDs, gathers the associated sample IDs, sorts them alphabetically,
# and records events for different CN types within these samples.
# The result is a series of output files, each containing event information for specific CN types
# within distinct clusters.
#
# Args:
# - clusterIDs (list[strs]): cluster IDs for which event data needs to be processed and saved.
# - samp2clusts (dict): maps sample IDs to their associated clusters
# - CNStates (list[strs]): CN types IDs
# - samp2CNEx_A (dict): maps CN states exon path for each sample, especially focusing on autosomal events.
# - plotDir (str): The directory where the output files will be saved.
def saveEventByCluster(clusterIDs, samp2clusts, CNStates, samp2CNEx_A, plotDir):
    for clusterID in clusterIDs:
        # Create a list of sample IDs associated with the current cluster
        sampList = [sampID for sampID, clust in samp2clusts.items() if clust[0] == clusterID]

        # Sort the sample IDs alphabetically
        sampList.sort()

        if not sampList:
            continue  # Skip empty clusters

        for CNtype in range(len(CNStates)):
            if CNtype == 2:  # Skip CNType 2
                continue

            # Initialize an empty dictionary to store cluster event information
            # keys = exonIndex[int], values = np.ndarray[NBsampInClust][bool]
            clustEventInd = {}

            for sampID in sampList:
                if sampID not in samp2CNEx_A:
                    continue

                # Get the CNsPath associated with the sample ID
                CNsPath = samp2CNEx_A[sampID]
                if CNsPath is not None:
                    # Find the event indexes matching the current CNType
                    eventInd = np.where(CNsPath == CNtype)[0]

                    for event in eventInd:
                        clustEventInd.setdefault(event, np.zeros(len(sampList), dtype=int))
                        clustEventInd[event][sampList.index(sampID)] = 1

            testFile = os.path.join(plotDir, f"Event_CN{CNtype}_{clusterID}.tsv")
            with open(testFile, "x") as outFH:
                header = "exonIndex\t" + "\t".join(sampList) + "\n"
                outFH.write(header)
                for i, counts in clustEventInd.items():
                    toPrint = [i] + counts
                    row2Print = '\t'.join(map(str, toPrint))
                    outFH.write(row2Print + "\n")


############################################
# processCluster
# Generate plots for clusters and CN states.

# Args:
# - clust2Counts (dict): keys = cluster IDs and values = CN count data as numpy arrays.
# - CNStates (list[strs]): CN state labels (e.g., ["CN0", "CN1", "CN2", "CN3"]).
# - pdf (PdfPages): A PDF object for saving the generated plots.
def processCluster(clust2Counts, CNStates, pdf):
    # Iterate through clusters in alphabetical order
    for cluster_id in sorted(clust2Counts.keys()):
        counts_list = np.array(clust2Counts[cluster_id])

        for CNstate in CNStates:
            CNcountsCurrent = counts_list[:, CNStates.index(CNstate)]

            if np.all(CNcountsCurrent == 0):
                logger.debug(f"Cluster {cluster_id} contains no CN counts for {CNstate}")
                continue

            # Calculate statistics
            mean = np.mean(CNcountsCurrent)
            median = np.median(CNcountsCurrent)
            stdev = np.std(CNcountsCurrent)

            # Create a new figure
            fig, ax = matplotlib.pyplot.subplots()

            try:
                histogramAndDensities(ax, CNcountsCurrent, f'{cluster_id} counts', 'skyblue')
            except Exception as e:
                logger.error(repr(e))

            # Plot mean and median lines
            ax.axvline(mean, color='grey', label=f'mean={mean:.3f}')
            ax.axvline(median, color='green', label=f'median={median:.3f}')

            # Plot standard deviation lines (conditional)
            if CNStates.index(CNstate) != 2:
                ax.axvline((mean + stdev), linestyle='--', color='grey', label=f'stdev=mean+{stdev:.3f}')
            else:
                ax.axvline((mean - stdev), linestyle='--', color='grey', label=f'stdev=mean-{stdev:.3f}')

            if CNStates.index(CNstate) == 2:
                ax.set_xlim(max(CNcountsCurrent) / 1.1, max(CNcountsCurrent) * 1.1)

            # Set labels and title
            ax.set_xlabel('CN exon counts')
            ax.set_ylabel('Densities')
            ax.set_title(f'Cluster {cluster_id} - NBsamps {len(CNcountsCurrent)} - CNState {CNstate}')
            ax.legend(loc='upper right', fontsize='small')

            pdf.savefig()
            matplotlib.pyplot.close(fig)


##############################
# histogramAndDensities
# Plot a histogram and density curve on the same axis.
#
# Args:
# - ax (matplotlib.axis): The matplotlib axis where the plot will be created.
# - data (array-like): Data for which the histogram and density are computed and plotted.
# - label (str): A label for the histogram and density curve.
# - color (str): The color for the histogram bars and density curve.
def histogramAndDensities(ax, data, label, color):
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
