import logging
import numpy as np
import os
import matplotlib.pyplot
import matplotlib.backends.backend_pdf

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################
# getTransitionMatrix
# Given a matrix that contains likelihood values for each copy number status
# per exon per sample, along with prior information on the copy numbers,
# this function performs the following steps:
# 1. Generates an observation matrix for copy numbers by combining the likelihood values and priors.
# 2. Calculates the observation lists for each sample by counting the transitions between copy number states.
#    It disregards "no-call" exons and the distances between exons during the counting process.
# 3. Creates a transition matrix based on the median values of the transition matrices from all evaluated samples.
# 4. Counts the number of copy number states obtained for each sample.
# 5. This count is used in debug mode to plot a bar plot, visualizing the frequency of each copy number.
#
# Args :
# - callsArrays (np.ndarray[floats]): likelihoods for each CN, each exon, each sample
#                                     dim = [nbExons, (nbSamps * nbCNStates)]
# - priors (np.ndarray[floats]): Prior probabilities for each copy number status.
# - samples (list[str]): sample names
# - CNStatus (list[str]): Names of copy number types.
# - outFolder [str] : path to save the graphical representation.
#
# Returns a tuple (obsDataBased, transtionMatrix):
# - obsDataBased (np.ndarray[int]) : observation lists for each sample (conserved no call as -1).
# - transtionMatrix (np.ndarray[floats]): median transition matrix.
def getTransitionMatrix(callsArrays, priors, samples, CNStatus, outFolder):

    obsDataBased = probs2CN(callsArrays, samples, priors, len(CNStatus), reverse_order=False)

    (countArray, transitionMatrices) = countCNObservations(obsDataBased, len(CNStatus))

    transtionMatrix = mergeTransMatrices(transitionMatrices)

    # Exclude the "filtered" category from the countArray
    countArray_no_filter = countArray[:, :-1]

    barPlot(countArray_no_filter, CNStatus, outFolder)

    return (obsDataBased, transtionMatrix)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#############################
# probs2CN
# Compute the observations of copy number (CN) states for each exon in each sample,
# considering both forward and backward directions of exon reading.
#
# Args:
# - CNcalls (ndarray[floats]): Array of shape (nbExons, nbSamples * nbStates) containing the
#                              probabilities of CN states for each exon and sample.
# - samples (list): List of sample names or indices.
# - priors (ndarray): Array of shape (nbStates,) containing the weights for calculating
#                     pseudoLogOdds.
# - nbStates (int): Number of CN states.
#
# Returns:
# - ndarray: Array of shape (nbExons, nbSamples) containing the observed CN states for
#     each exon and sample. The observed CN state is represented as an index.
def probs2CN(CNcalls, samples, priors, nbStates):
    nbSamples = len(samples)

    # List to store CN observations for each sample
    observations_list = []

    # Set to store filtered sample indices
    filtered_samples = set()

    # Check if the entire probsArray is -1 for each sample
    for sampleIndex in range(nbSamples):
        probsArray = CNcalls[:, sampleIndex * nbStates:sampleIndex * nbStates + nbStates]
        if np.all(probsArray == -1):
            filtered_samples.add(sampleIndex)
            # Print the names of filtered samples
            print("Filtered samples:", samples[sampleIndex])
        else:
            # Continue with the observation calculations for non-filtered samples
            # Calculate the weighted probabilities using probsArray and priors.
            # The multiplication of probsArray and priors represents a weighted version
            # of the probabilities, where probsArray contains the probabilities of events
            # and priors contains the corresponding weights.
            # This calculation is analogous to the odds ratio calculation in statistics,
            # where the odds ratio is defined as the ratio of the odds of two events.
            # The odds ratio captures the relationship between the weighted probabilities.
            # Specifically, the odds ratio is given by:
            #   OddsRatio = (P(A) * (1 - P(B))) / (P(B) * (1 - P(A)))
            # where P(A) and P(B) are the weighted probabilities of events A and B, respectively.
            odds = probsArray * priors
            observations = np.argmax(odds, axis=1)
            # Check if any element in the row is -1
            any_minus_one = np.any(probsArray == -1, axis=1)
            observations[any_minus_one] = -1
            observations_list.append(observations)

    # Concatenate observations for non-filtered samples
    observations = np.vstack(observations_list)

    # Transpose the observations array
    observations = observations.transpose()

    return observations


#############################
# countCNObservations
# Count the observations of copy number (CN) states for each patient and CN type,
# and calculate the transition matrices.
#
# Args:
# - observations (np.ndarray[int]): Array of shape (nbExons, nbSamples) containing the observed CN states
#     for each exon and sample.
# - nbStates [int]: Number of CN states.
#
# Returns:
# tuple: A tuple containing:
# - ndarray: Array of shape (nbSamples, nbStates) containing the count of each CN type for each sample.
# - list: List of transition matrices, each of shape (nbStates, nbStates), representing the transitions
#     between CN types for each sample.
def countCNObservations(observations, nbStates):
    nbExons, nbSamples = observations.shape

    # Array to store the count of each CN type for each sample
    countArray = np.zeros((nbSamples, nbStates), dtype=int)

    # List to store the transition matrices
    transitionMatrices = []

    for sampleIndex in range(nbSamples):
        if np.all(observations[:, sampleIndex] != -1):
            continue

        # Count occurrences of each CN type, excluding -1 values
        valid_observations = observations[:, sampleIndex][observations[:, sampleIndex] != -1]
        counts = np.bincount(valid_observations, minlength=nbStates)
        countArray[sampleIndex] = counts

        # Calculate transition matrix
        countMatrix = np.zeros((nbStates, nbStates), dtype=int)

        prevCN = 2
        for exonIndex in range(nbExons):
            currCN = observations[exonIndex, sampleIndex]
            if currCN == -1:
                continue
            countMatrix[prevCN, currCN] += 1
            prevCN = currCN

        transitionMatrices.append(countMatrix)

    return (countArray, transitionMatrices)


######################################################
# mergeTransMatrices
# Merges a list of transition matrices by taking the median of each element in the list.
# The resulting matrix is then normalized, ensuring that the sum of each row equals 1.
#
# Args:
# - transMatricesList (list[np.ndarray[int]]): List of transition matrices to be merged.
#
# Returns:
# - normalized_arr (np.ndarray[floats]): Merged and normalized transition matrix.
def mergeTransMatrices(transMatricesList):
    merged_array = np.median(transMatricesList, axis=0)

    # Normalize each row to ensure sum equals 1
    row_sums = np.sum(merged_array, axis=1, keepdims=True)
    normalized_arr = merged_array / row_sums

    return (normalized_arr)


#############################################################
# barPlot
# Creates a bar plot of copy number frequencies based on the count array.
# The plot includes error bars representing the standard deviation.
#
# Args:
# - countArray (np.ndarray[ints]): Count array representing copy number frequencies.
# - CNStatus (list[str]): Names of copy number states.
# - outFolder (str): Path to the output folder for saving the bar plot.
def barPlot(countArray, CNStatus, outFolder):
    pdf = os.path.join(outFolder, "CNFreqBarplot.pdf")
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf)
    fig = matplotlib.pyplot.figure(figsize=(8, 8))

    # Calculate the mean and standard deviation for each category
    means = np.mean(countArray, axis=0)
    stds = np.std(countArray, axis=0)

    # Normalize the means to get frequencies
    total_mean = np.sum(means)
    frequencies = means / total_mean

    # Plot the bar plot with error bars
    matplotlib.pyplot.bar(CNStatus, frequencies, yerr=stds / total_mean, capsize=3)

    # Define the vertical offsets for the annotations dynamically based on standard deviation
    mean_offset = np.max(frequencies) * 0.1
    std_offset = np.max(frequencies) * 0.05

    # Add labels for mean and standard deviation above each bar
    for i in range(len(CNStatus)):
        matplotlib.pyplot.text(i, frequencies[i] + mean_offset, f'mean={frequencies[i]:.2e}', ha='center')
        matplotlib.pyplot.text(i, frequencies[i] + std_offset, f'stdev={stds[i]/total_mean:.2e}', ha='center')

    # Set the labels and title
    matplotlib.pyplot.xlabel('Copy number States')
    matplotlib.pyplot.ylabel('Frequencies')
    matplotlib.pyplot.title(f'CN frequencies Bar Plot for {len(countArray)} samps (Excluding Filtered)')
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    matplotOpenFile.close()
