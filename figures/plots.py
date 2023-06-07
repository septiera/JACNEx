import os
import logging
import numpy as np
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
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
# visualisation of clustering results by a dendogram
# the labels below the figure correspond from bottom to top
# to the clusters formation in decreasing order of correlation
# label types:
# " ": sample does not contribute to the cluster
# "x": sample contributes to the cluster
# "-": sample controls the cluster
# "O": sample is not successfully clustered
#
# Args:
# - linksMatrix (np.ndarray[float])
# - labelsGp (list[str]): labels for each sample within each cluster, dim=NbSOIs*NbClusters
# - minDist (float): is the distance to start cluster construction
# - CM [str]: clustering method
# - pdf: matplotlib PDF object where the plot will be saved
#
# Returns a pdf file in the output folder
def plotDendogram(linksMatrix, labelsGp, minDist, CM, pdf):
    # Disable interactive mode
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(15, 5), facecolor="white")
    matplotlib.pyplot.title(CM + " linkage hierarchical clustering")
    dn1 = scipy.cluster.hierarchy.dendrogram(linksMatrix, labels=labelsGp, color_threshold=minDist)
    matplotlib.pyplot.ylabel("Distance √(1-ρ) ")
    fig.subplots_adjust(bottom=0.3)
    pdf.savefig(fig)
    matplotlib.pyplot.close()


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
