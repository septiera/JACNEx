import logging
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import numpy
import os
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
# visualize clustering results with a dendrogram.
# Most of the work is defining the leaf labels, which appear below the dendrogram,
# and the link colors, which should represent the constructed clusters.
# Args: see calling code in buildClusters()
# Produces plotFile (pdf format), returns nothing
def plotDendrogram(linkageMatrix, samples, clust2samps, fitWith, clustIsValid, title, plotFile):
    numSamples = len(samples)
    # build samp2clust for efficiency: key==sampleID, value==clusterID
    samp2clust = {}
    for clust in clust2samps.keys():
        for samp in clust2samps[clust]:
            samp2clust[samp] = clust

    ##################
    # leaf labels: we want to label the "middle" sample (visually in the dendrogram) of each
    # cluster with the clusterID eg 'A_02', other samples don't get labeled;
    # furthermore: if the cluster needs friends for fitting it gets lowercased eg 'a_02', and if it
    # is invalid it is lowercased and parenthesized eg '(a_02)'
    # key == sample index in samples (only defined for samples that get labeled), value == label
    sampi2label = {}
    # Also populate clust2color: key==clusterID, value == matplotlib color string code, we want to
    # cycle over a palette in the visual dendrogram order
    clust2color = {}
    allColors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    # next color index (in allColors)
    nextColi = 0

    # produce a first virtual dendrogram to get the order of samples/leaves
    R = scipy.cluster.hierarchy.dendrogram(linkageMatrix, get_leaves=True, count_sort='descending', no_plot=True)
    pos2si = R['leaves']
    # pos2si[pos] == si  <=> the leave at position pos (left-to-right, < numSamples) is sample index si

    nextPos = 0
    while nextPos < numSamples:
        nextSample = samples[pos2si[nextPos]]
        nextClust = samp2clust[nextSample]
        # color for nextClust
        clust2color[nextClust] = allColors[nextColi]
        nextColi += 1
        if nextColi == len(allColors):
            nextColi = 0
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
    # for trouble-shooting: log sampleIDs + clusterIDs from left to right in the dendrogram
    logger.debug("dendrogram leaves from left to right:")
    for pos in range(numSamples):
        sample = samples[pos2si[pos]]
        clust = samp2clust[sample]
        logger.debug("%s - %s", clust, sample)

    ##################
    # link colors: one color for each clusterID
    # for each non-singleton node i (== row index in linkageMatrix), populate node2clust:
    # node2clust[i] == the clusterID that node i belongs to if it belongs to a cluster,
    # or "V:clusterID" if node i is a "virtual merger" of several (unmerged) clusters including
    # clusterID (the choice of the clusterID to use is arbitrary and doesn't matter)
    node2clust = [None] * linkageMatrix.shape[0]

    # set n2c[c] = clust for non-singleton nodei and all its non-singleton descendants
    # whose pre-existing n2c value starts with 'V:'
    def setVirtualsToClust(n2c, nodei, clust):
        if n2c[nodei].startswith('V:'):
            n2c[nodei] = clust
            for ci in range(2):
                child = int(linkageMatrix[nodei, ci])
                if child >= numSamples:
                    setVirtualsToClust(n2c, child - numSamples, clust)

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

        if clust1.startswith('V:') and clust2.startswith('V:'):
            # both are virtual mergers, arbitrarily use the first
            node2clust[node] = clust1
        elif clust1.startswith('V:') or clust2.startswith('V:'):
            # switch if needed so c1 is the virtual
            if clust2.startswith('V:'):
                ctmp = c2
                c2 = c1
                c1 = ctmp
                ctmp = clust2
                clust2 = clust1
                clust1 = ctmp
            virtClust = clust1.removeprefix('V:')
            if virtClust in fitWith[clust2]:
                # node and all its 'V:' descendants are "given" to clust2 (for coloring)
                node2clust[node] = clust2
                setVirtualsToClust(node2clust, c1 - numSamples, clust2)
            else:
                # define node as Virtual
                node2clust[node] = clust1
        else:
            # c1 and c2 are both non-virtual
            if (clust1 == clust2):
                node2clust[node] = clust1
            elif (clust2 in fitWith[clust1]):
                node2clust[node] = clust1
            elif (clust1 in fitWith[clust2]):
                node2clust[node] = clust2
            else:
                node2clust[node] = 'V:' + clust1

    # finally we can define the node/link color function
    def linkColorFunc(node):
        node -= numSamples
        if node2clust[node].startswith('V:'):
            # node doesn't belong to a non-virtual cluster, default to black
            return('k')
        else:
            return(clust2color[node2clust[node]])

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


def plotExon(BLABLABLA, TODO):
    return()


#############################################################
# plotPieChart:
# Generate and save a pie chart representing the distribution of exon status
# (filtered or called).
#
# Args:
# - clustID [str]: cluster identifier
# - filterStates (list[strs]): filter states IDs
# - exStatusArray (numpy.ndarray[ints]): exon filtering states indexes
# - plotDir: Folder path for save the generated plot
def plotPieChart(clustID, filterStates, exStatusArray, plotDir):
    name_pie_charts_file = f"exonsFiltersSummary_pieChart_cluster_{clustID}.pdf"
    pdf_pie_charts = os.path.join(plotDir, name_pie_charts_file)
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf_pie_charts)

    # Use numpy.unique() to obtain unique values and their occurrences
    # with return_counts=True
    uniqueValues, counts = numpy.unique(exStatusArray[exStatusArray != -1], return_counts=True)

    # Create the pie chart figure and subplot
    fig = matplotlib.pyplot.figure(figsize=(5, 5))
    ax11 = fig.add_subplot(111)

    # Plot the pie chart with customization
    w, l, p = ax11.pie(counts,
                       labels=None,
                       autopct=lambda x: str(round(x, 2)) + '%',
                       textprops={'fontsize': 14},
                       startangle=160,
                       radius=0.5,
                       pctdistance=1,
                       labeldistance=None)

    # Calculate percentage distances for custom label positioning
    step = (0.8 - 0.2) / (len(filterStates) - 1)
    pctdists = [0.8 - i * step for i in range(len(filterStates))]

    # Position the labels at custom percentage distances
    for t, d in zip(p, pctdists):
        xi, yi = t.get_position()
        ri = numpy.sqrt(xi**2 + yi**2)
        phi = numpy.arctan2(yi, xi)
        x = d * ri * numpy.cos(phi)
        y = d * ri * numpy.sin(phi)
        t.set_position((x, y))

    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.title("Filtered and called exons from cluster " + str(clustID))
    matplotlib.pyplot.legend(loc='upper right', fontsize='small', labels=filterStates)
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    matplotOpenFile.close()


#############################################################
# barPlot
# Creates a bar plot of copy number frequencies based on the count array.
# The plot includes error bars representing the standard deviation.
#
# Args:
# - countArray (numpy.ndarray[ints]): Count array representing copy number frequencies.
# - CNStatus (list[str]): Names of copy number states.
# - outFolder (str): Path to the output folder for saving the bar plot.
def barPlot(countArray, CNStatus, pdf):
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf)
    fig = matplotlib.pyplot.figure(figsize=(10, 8))

    # Calculate the mean and standard deviation for each category
    means = numpy.mean(countArray, axis=0)
    stds = numpy.std(countArray, axis=0)

    # Normalize the means to get frequencies
    total_mean = numpy.sum(means)
    frequencies = means / total_mean

    # Plot the bar plot with error bars
    matplotlib.pyplot.bar(CNStatus, frequencies, yerr=stds / total_mean, capsize=3)

    # Define the vertical offsets for the annotations dynamically based on standard deviation
    mean_offset = numpy.max(frequencies) * 0.1
    std_offset = numpy.max(frequencies) * 0.05

    # Add labels for mean and standard deviation above each bar
    for i in range(len(CNStatus)):
        matplotlib.pyplot.text(i, frequencies[i] + mean_offset, f'μ={frequencies[i]:.1e}', ha='center')
        matplotlib.pyplot.text(i, frequencies[i] + std_offset, f'σ={stds[i] / total_mean:.1e}', ha='center')

    # Set the labels and title
    matplotlib.pyplot.xlabel('Copy number States')
    matplotlib.pyplot.ylabel('Frequencies')
    matplotlib.pyplot.title(f'CN frequencies Bar Plot for {len(countArray)} samps (Excluding Filtered)')
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    matplotOpenFile.close()
