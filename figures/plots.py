import logging
import numpy as np
import matplotlib.pyplot
import scipy.cluster.hierarchy

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# Plot one or more curves, and optionally two vertical dashed lines, on a
# single figure.
# Each curve is passed as an ndarray of X coordinates (eg dataRanges[2] for the
# third curve), a corresponding ndarray of Y coordinates (densities[2]) of the
# same length, and a legend (legends[2]).
# The vertical dashed lines are drawn at X coordinates line1 and line2, unless
# line1==line2==0.
#
# Args:
# - title: plot's title (string)
# - dataRanges: list of N ndarrays storing X coordinates
# - densities: list of N ndarrays storing the corresponding Y coordinates
# - legends: list of N strings identifying each (dataRange,density) pair
# - line1, line2 (floats): X coordinates of dashed vertical lines to draw
# - line1legend, line2legend (strings): legends for the vertical lines
# - ylim (float): Y max plot limit
# - pdf: matplotlib PDF object where the plot will be saved
#
# Returns a pdf file in the output folder
def plotDensities(title, dataRanges, densities, legends, line1, line2, line1legend, line2legend, ylim, pdf):
    # sanity
    if (len(dataRanges) != len(densities)) or (len(dataRanges) != len(legends)):
        raise Exception('plotDensities bad args, length mismatch')

    # set X max plot limit (both axes start at 0)
    xlim = max(dataRanges[:][-1])

    # Disable interactive mode
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(6, 6))
    for i in range(len(dataRanges)):
        matplotlib.pyplot.plot(dataRanges[i], densities[i], label=legends[i])

    if (line1 != 0) or (line2 != 0):
        matplotlib.pyplot.axvline(line1, color='crimson', linestyle='dashdot', linewidth=1, label=line1legend)
        matplotlib.pyplot.axvline(line2, color='darkorange', linestyle='dashdot', linewidth=1, label=line2legend)

    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("density")
    matplotlib.pyplot.xlim(0, xlim)
    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc='upper right', fontsize='small')

    pdf.savefig(fig)
    matplotlib.pyplot.close()


#############################
# visualisation of clustering results
# Args:
# - clust2Samps (dict(int : list[int])): clusterID associated to valid sample indexes
#   key = clusterID, value = list of valid sample indexes
# - trgt2Ctrls (dict(int : list[int])): target and controls clusters correspondance,
#   key = target clusterID, value = list of controls clusterID
# - linksMatrix (np.ndarray[float])
# - minDist (float): is the distance to start cluster construction
# - outputFile (str): full path to save the png
# Returns a png file in the output folder
def plotDendogram(clusters, samples, linksMatrix, minDist, outputFile):
    # maxClust: int variable contains total clusters number
    maxClust = len(clusters)

    # To Fill
    # labelArray (np.ndarray[str]): status for each cluster as a character, dim=NbSOIs*NbClusters
    # " ": sample does not contribute to the cluster
    # "x": sample contributes to the cluster
    # "-": sample controls the cluster
    # "O": sample not clustered
    labelArray = np.empty([len(samples), maxClust + 1], dtype="U1")
    labelArray.fill(" ")
    # labelsGp (list[str]): labels for each sample list to be passed when plotting the dendogram
    labelsGp = []

    # browse the different cluster identifiers
    for i in range(len(clusters)):
        # retrieving the samples involved for the clusterID
        sampsNames = "".join(clusters[i][1]).split(",")
        sampsIndex = [samples.index(sampsName) for sampsName in sampsNames]

        if clusters[i][0] == "Samps_ClustFailed":
            labelArray[sampsIndex, i] = "O"
            continue

        # associate the label for the samples contributing to the clusterID for the
        # associated cluster index position
        labelArray[sampsIndex, i] = "x"

        # associate the label for the samples controlling the current clusterID
        if clusters[i][2] != "":
            listctrl = [x.strip() for x in clusters[i][2].split(",")]
            for ctrl in listctrl:
                CTRLclustInfo = [sublist[1] for sublist in clusters if sublist[0] == ctrl]
                CTRLsampsNames = "".join(CTRLclustInfo).split(",")
                CTRLsampsIndex = [samples.index(sampsName) for sampsName in CTRLsampsNames]
                labelArray[CTRLsampsIndex, i] = "-"

    # browse the np array of labels to build the str list
    for i in labelArray:
        # separation of labels for readability
        strToBind = "  ".join(i)
        labelsGp.append(strToBind)

    # dendogram plot
    matplotlib.pyplot.figure(figsize=(15, 5), facecolor="white")
    matplotlib.pyplot.title("Average linkage hierarchical clustering")
    dn1 = scipy.cluster.hierarchy.dendrogram(linksMatrix, labels=labelsGp, color_threshold=minDist)
    matplotlib.pyplot.ylabel("Distance √(1-ρ) ")
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="pdf", bbox_inches='tight')
