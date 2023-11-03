import logging
import numpy as np
import os
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import datetime

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
#######################
# getCNsPaths
# Compute copy number states (CNs) paths based on likelihoods and priors.
#
# Args:
# - likelihoodDict (dict): key==sampleIDs, value==np.array2D of likelihoods.
#                          dim = nbExons * nbCNStates
# - priors (list[floats]): Prior probabilities for each CN states.
#
# Returns a dictionary mapping sample IDs to CNs paths[ints], where -1
# represents no calls exons.
def getCNsPaths(likelihoodDict, priors):   
    sampCNPathDict = {}

    # Iterate through each sample (sampID) and its likelihoods
    for sampID, likelihoods in likelihoodDict.items():
        # Create a boolean mask to identify positions with non -1
        # values (no calls)
        mask = (likelihoods != -1)
        odds = likelihoods.copy()  # avoid modifying the original data
        odds[mask] = odds[mask] * priors

        # Initialize CNsList with -1 values for all positions
        CNsList = np.full(likelihoods.shape[0], -1, dtype=int)
        CNsList[mask] = np.argmax(odds[mask], axis=1)
        sampCNPathDict[sampID] = CNsList

    return sampCNPathDict


#######################
# countCNPerSample
# Count Copy Number (CN) occurrences per sample.
#
# Args:
# - sampCNPathDict (dict): key==sampleIDs, value== numpy array1D (nbExons) of CNs[ints]
#
# Returns a dictionary where keys are sample IDs, and values are
# NumPy arrays containing CN counts per sample.
def countCNPerSample(sampCNPathDict):
    sampCNCountsDict = {}
    for sampID, CNpath in sampCNPathDict.key():
        # Filter out the values of -1
        filtered_arr = CNpath[CNpath != -1]

        # Use np.bincount to count the occurrences of each value
        counts = np.bincount(filtered_arr)

        sampCNCountsDict[sampID] = counts
    
    return sampCNCountsDict


#########################
# identifyOutliers
# Identify outliers in cluster-based sample counts and generate histograms.
#
# Args:
# - clusterDict (dict):key==clusterIDs, value==lists of sample IDs[strs].
# - sampCNCountsDict(dict): key==sampleIDs, value==1D arrays of sample
#                           counts by CNs[ints].
# - plotDir [str]: The directory where histograms will be saved.
# - threshold (int, optional): The number of standard deviations from the mean
#                              to consider a sample an outlier. Default is 3.
#
# Returns a dictionary: keys = cluster IDs, and values = sets of outlier sample IDs.
def identifyOutliers(clusterDict, sampCNCountsDict, plotDir, threshold=3):
    # init data structure
    outliers = {}  # output dictionary
    # debug pdf of histograms
    date = datetime.datetime.now().strftime('%y%m%d')
    histogramsFile = os.path.join(plotDir, f"CNcountsHistograms_{date}.pdf")
    backendPDF = matplotlib.backends.backend_pdf.PdfPages(histogramsFile)

    # Iterate through clusters and their respective sample IDs
    for cluster, sampleIDs in clusterDict.items():
        # Collect sample counts for the current cluster
        clusterCounts = [sampCNCountsDict[sampleID] for sampleID in sampleIDs]
        clusterCounts = np.array(clusterCounts)

        # Calculate mean and standard deviation for each cluster
        mean = np.mean(clusterCounts, axis=0)
        std_dev = np.std(clusterCounts, axis=0)

        # Calculate the upper threshold for each column
        upper_threshold = mean + threshold * std_dev

        ###########################
        #### DEBUG : Generate histograms and add them to the PDF file
        try:
            plotHistogramsAndStats(clusterCounts, mean, upper_threshold, cluster, backendPDF)
        except Exception as e:
            logger.error(f"plotHistogramsAndStats failed for cluster {cluster}")
        ###########################

        # Identify and store outlier samples
        outlier_samples = set()
        for i, column_counts in enumerate(clusterCounts.T):
            for sampleID, count in zip(sampleIDs, column_counts):
                if count > upper_threshold[i]:
                    # Ignore CN2 (index 2) and add only if i is not 2
                    if i != 2:
                        log_message = f"Sample {sampleID} in cluster {cluster} is an outlier for CN{i} (mean: {mean[i]}, std_dev: {std_dev[i]}, counts value: {count})"
                        logger.warning(log_message)
                        outlier_samples.add(sampleID)

        outliers[cluster] = outlier_samples

    backendPDF.close()
    return(outliers)

#######################
# plotHistogramsAndStats DEBUG function
# Plot histograms and statistics for a cluster's copy number counts.
#
# Args:
# - clusterCounts (numpy.ndarray): 2D array containing copy number counts
#                                  for each sample in a cluster
# - mean (numpy.ndarray): mean values for each CNs.
# - upper_threshold (numpy.ndarray): upper threshold values for each CNs.
# - cluster [str]
# - backendPDF (matplotlib.backends.backend_pdf.PdfPages): PDF backend for saving plots.
def plotHistogramsAndStats(clusterCounts, mean, upper_threshold, cluster, backendPDF):
    for i, column_counts in enumerate(clusterCounts.T):
        if i != 2:  # no plot for CN2
            matplotlib.pyplot.figure(figsize=(8, 6))
            matplotlib.pyplot.hist(column_counts, bins=len(column_counts)/2, alpha=0.5, color='b', label='Counts')
            matplotlib.pyplot.axvline(mean[i], color='r', linestyle='dashed', linewidth=2, label='Mean')
            matplotlib.pyplot.axvline(upper_threshold[i], color='g', linestyle='dashed', linewidth=2, label='Upper Threshold')
            matplotlib.pyplot.title(f'Cluster {cluster} - CN{i} Histogram')
            matplotlib.pyplot.xlabel('Counts')
            matplotlib.pyplot.ylabel('Frequency')
            matplotlib.pyplot.legend()
            backendPDF.savefig()

#######################
# updateClusteringData
# Updates clustering data by creating new clusters for outliers and removing
# those outliers from the existing clusters.
#
# Args:
# - clust2samps (dict): key==clusterID, value== list of sampleIDs
# - outliers (dict): key==clusterIDs, value==lists of outlier sampleIDs.
# - fitWith (dict): keys==clusterIDs, value==lists of cluster IDs to fit with.
# - clustIsValid (dict): keys==clusterIDs, value==Boolean values indicating cluster validity.
# - minSamps [int]:  
#
# Returns:
# - Tuple: A tuple containing updated dictionaries for clust2samps, fitWith, and clustIsValid.
def updateClusteringData(clust2samps, outliers, fitWith, clustIsValid, minSamps):
    # Create new dictionaries to store the updated data
    newClust2Samps = {}
    newFitWith = {}
    newClustIsValid = {}

    # Iterate through the existing clusters
    for clusterID, sampleIDs in clust2samps.items():
        # Check if the cluster has outliers
        if clusterID in outliers:
            outlierIDs = outliers[clusterID]  # Get the IDs of the dubious samples
            if outlierIDs:
                # Generate a new name for the cluster
                newCluster = generateNewClusterID(clusterID, clust2samps, newClust2Samps)

                # Create a new cluster with the dubious samples
                newClust2Samps[newCluster] = list(outlierIDs)

                # Remove the dubious samples from the old cluster sample list
                newSampleIDs = [sampleID for sampleID in sampleIDs if sampleID not in outlierIDs]
                newClust2Samps[clusterID] = newSampleIDs

                # Update fitWith and clustIsValid for the old and new clusters
                newFitWith[newCluster] = []
                newClustIsValid[newCluster] = False
                newClustIsValid[clusterID] = (len(newSampleIDs) >= minSamps)  

        else:
            # Add the existing cluster to the new dictionaries
            newClust2Samps[clusterID] = sampleIDs
            newFitWith[clusterID] = fitWith[clusterID]
            newClustIsValid[clusterID] = clustIsValid[clusterID]

    return (newClust2Samps, newFitWith, newClustIsValid)


#######################
# generateNewClusterID 
# Generate a new cluster ID based on the existing cluster IDs while
# considering group name and numbering.
#
# Args:
# - clusterID (str): The base name of the cluster.
# - clust2samps (dict): key==clusterID, value== list of sampleIDs
# - newClust2samps (dict): update version of clust2samps
#
# Returns:
# str: A new cluster ID that follows the group name and numbering pattern.
def generateNewClusterID(clusterID, clust2samps, newClust2samps):
    # Determine the group name ("A" or "G") from the baseName
    group_name = clusterID.split('_')[0]

    # Combine existing clusters and new clusters for checking uniqueness
    all_clusters = set(list(clust2samps.keys()) + list(newClust2samps.keys()))

    # Find the maximum group number among existing clusters and new clusters
    # with the same group name
    max_group_number = 0
    for cluster in all_clusters:
        if cluster.startswith(group_name):
            cluster_parts = cluster.split('_')
            if len(cluster_parts) == 2 and cluster_parts[1].isdigit():
                max_group_number = max(max_group_number, int(cluster_parts[1]))

    # Increment the group number
    new_group_number = max_group_number + 1

    # Generate the new cluster name
    newClusterName = f"{group_name}_{new_group_number:02}"

    return newClusterName


