import os
import logging
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import scipy.cluster.hierarchy

# prevent matplotlib and PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
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
# Returns nothing.
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
# visualize clustering results with a dendrogram
# Args: see calling code in buildClusters()
# Returns nothing
def plotDendrogram(linkageMatrix, samples, clust2samps, fitWith, clustIsValid, title, plotFile):
    ##################
    # most of the work here is defining the leaf labels, which appear below the dendrogram:
    # we want to label the "middle" sample (visually in the dendrogram) of each cluster with
    # the clusterID eg 'A_02', other samples don't get labeled;
    # furthermore: if the cluster needs friends for fitting it gets lowercased eg 'a_02', and if it
    # is invalid it is lowercased and parenthesized eg '(a_02)'
    numSamples = len(samples)
    sampi2label = {}
    # produce a first virtual dendrogram to get the order of samples/leaves
    R = scipy.cluster.hierarchy.dendrogram(linkageMatrix, get_leaves=True, count_sort='descending', no_plot=True)
    pos2si = R['leaves']
    # pos2si[pos] == si  <=> the leave at position pos (left-to-right, < numSamples) is sample index si

    # build samp2clust for efficiency: key==sampleID, value==clusterID
    samp2clust = {}
    for clust in clust2samps.keys():
        for samp in clust2samps[clust]:
            samp2clust[samp] = clust

    nextPos = 0
    while nextPos < numSamples:
        nextSample = samples[pos2si[nextPos]]
        nextClust = samp2clust[nextSample]
        # leaves from nextPos to nextPos+len(clust2samps[nextClust])-1 belong to cluster nextClust, label
        # the leaf in the middle
        posToLabel = nextPos + (len(clust2samps[nextClust]) - 1) // 2

        # create label for this cluster, "decorated" if it needs friends / is invalid
        label = ""
        if not clustIsValid[nextClust]:
            label = '(' + nextClust.lower() + ')'
        elif len(fitWith[nextClust]) > 0:
            label = nextClust.lower()
        else:
            label = nextClust

        sampi2label[pos2si[posToLabel]] = label
        nextPos += len(clust2samps[nextClust])

    # finally we can define the leaf label function
    def leafLabelFunc(sampi):
        if sampi in sampi2label:
            return sampi2label[sampi]
        else:
            return('')

    ##################
    # we also want to define the link colors: one color for each clusterID
    # for each non-singleton node i (== row index in linkageMatrix), populate node2clust:
    # node2clust[i] == the clusterID that node i belongs to (if it belongs to a cluster, stays False otherwise)
    node2clust = [None] * linkageMatrix.shape[0]
    for node in range(linkageMatrix.shape[0]):
        (c1, c2, dist, size) = linkageMatrix[node]
        c1 = int(c1)
        c2 = int(c2)
        if c1 < numSamples:
            clust1 = samp2clust[samples[c1]]
        else:
            clust1 = node2clust[c1 - numSamples]
        if c2 < numSamples:
            clust2 = samp2clust[samples[c2]]
        else:
            clust2 = node2clust[c2 - numSamples]
        if clust1 and (clust1 == clust2):
            node2clust[node] = clust1
    # pick a color for each cluster: key==clusterID, value==matplotlib color string code
    clust2color = {}
    allColors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    coli = 0
    for clust in clust2samps.keys():
        clust2color[clust] = allColors[coli]
        coli += 1
        if coli >= len(allColors):
            coli = 0

    # finally we can define the node/link color function
    def linkColorFunc(node):
        node -= numSamples
        if node2clust[node]:
            return(clust2color[node2clust[node]])
        else:
            # node isn't in any cluster, default to black
            return('k')

    ##################
    # make the plot
    pdf = matplotlib.backends.backend_pdf.PdfPages(plotFile)
    # Disable interactive mode
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(25, 10))
    matplotlib.pyplot.title(title)
    scipy.cluster.hierarchy.dendrogram(linkageMatrix,
                                       leaf_rotation=30,
                                       leaf_label_func=leafLabelFunc,
                                       link_color_func=linkColorFunc,
                                       count_sort='descending')

    matplotlib.pyplot.ylabel("Distance")
    pdf.savefig(fig)
    matplotlib.pyplot.close()
    pdf.close()


#############################################################
# plotPieChart:
# generates a pie by cluster resuming the filtering of exons
#
# Args:
# - clustID [str]: cluster identifier
# - filterCounters (dict[str:int]): dictionary of exon counters of different filtering
# performed for the cluster
# - pdf [str]: file path to save plot
#
# save a plot in the output pdf
def plotPieChart(clustID, filterCounters, pdf):
    
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf)
    
    fig = matplotlib.pyplot.figure(figsize=(5, 5))
    ax11 = fig.add_subplot(111)
    w, l, p = ax11.pie(filterCounters.values(),
                       labels=None,
                       autopct=lambda x: str(round(x, 2)) + '%',
                       textprops={'fontsize': 14},
                       startangle=160,
                       radius=0.5,
                       pctdistance=1,
                       labeldistance=None)

    step = (0.8 - 0.2) / (len(filterCounters.keys()) - 1)
    pctdists = [0.8 - i * step for i in range(len(filterCounters.keys()))]

    for t, d in zip(p, pctdists):
        xi, yi = t.get_position()
        ri = np.sqrt(xi**2 + yi**2)
        phi = np.arctan2(yi, xi)
        x = d * ri * np.cos(phi)
        y = d * ri * np.sin(phi)
        t.set_position((x, y))

    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.title("Filtered and called exons from cluster " + str(clustID))
    matplotlib.pyplot.legend(loc='upper right', fontsize='small', labels=filterCounters.keys())
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    
    matplotOpenFile.close()


#########################
# plotExponentialFit
# generates a plot to visualize the exponential fit for a given dataset.
# It takes the plot title, x-axis data, y-axis data, plot legends, and the
# output file path for saving the plot as input.
#
# Args:     
# - plotTitle [str]: The title of the plot.
# - xi (np.ndarray[floats]): The x-axis data (FPM values).
# - yLists (np.ndarray[floats]): A list of y-axis data for plotting.
# - plotLegs (list[str]): A list of legends for each dataset.
# - pdf [str]: The output file path for saving the plot as a PDF.
#
# save a plot in the output pdf
def plotExponentialFit(plotTitle, dr, yLists, plotLegs, pdf):
    # sanity
    if (len(yLists[0]) != len(yLists[1])) or (len(yLists) != len(plotLegs)):
        raise Exception('plotDensities bad args, length mismatch')
    
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf)
    fig = matplotlib.pyplot.figure(figsize=(5, 5))
    
    matplotlib.pyplot.plot(dr, yLists[0], label=plotLegs[0])
    matplotlib.pyplot.plot(dr, yLists[1], label=plotLegs[1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title(plotTitle)
    matplotlib.pyplot.xlabel('FPM')
    matplotlib.pyplot.ylabel('densities')
    matplotlib.pyplot.ylim(0, max(yLists[0])/100)
    matplotlib.pyplot.xlim(0, max(dr)/3)
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    matplotOpenFile.close()
    
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

    # Define a list of colours based on the number of distributions to plot.
    # The 'plasma' colormap is specifically designed for people with color vision deficiencies.
    distColor = matplotlib.pyplot.cm.get_cmap('plasma', len(xi))
    vertColor = matplotlib.pyplot.cm.get_cmap('plasma', len(verticalLines))

    # Disable interactive mode to prevent display of the plot during execution
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(8, 8))
    # Plot a density histogram of the raw data with a number of bins equal to half the number of data points
    matplotlib.pyplot.hist(rawData, bins=int(len(rawData) / 2), density=True)

    # Plot the density/distribution curves for each set of x- and y-values
    if len(yLists) > 1:
        for i in range(len(yLists)):
            # Choose a color based on the position of the curve in the list
            color = distColor(i / len(yLists))
            matplotlib.pyplot.plot(xi, yLists[i], color=color, label=plotLegs[i])

    # Plot vertical lines to mark points of interest on the histogram
    if verticalLines:
        for j in range(len(verticalLines)):
            color = vertColor(j / len(verticalLines))
            matplotlib.pyplot.axvline(verticalLines[j], color=color, linestyle='dashdot', linewidth=1, label=vertLinesLegs[j])

    # Set the x- and y-axis labels, y-axis limits, title, and legend
    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("Density or PDF")
    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.title(plotTitle)
    matplotlib.pyplot.legend(loc='upper right', fontsize='small')

    pdf.savefig(fig)
    matplotlib.pyplot.close()
