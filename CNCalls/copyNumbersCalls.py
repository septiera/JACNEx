import os
import logging
import numpy as np
import numba
import scipy.stats
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import time

import clusterSamps.smoothing
import clusterSamps.genderDiscrimination

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#####################################
# allocateLogOddsArray:
# Args:
# - exons (dict[str, List[int]]): key: clusterID , value: samples index list
# - SOIs (np.ndarray[float]): normalised counts
# Return:
# - Returns an all zeroes float array, adapted for
# storing the logOdds for each type of copy number.
# dim= NbExons x [NbSOIs x [CN0, CN1, CN2,CN3+]]
def allocateLogOddsArray(SOIs, exons):
    # order=F should improve performance
    return (np.zeros((len(exons), (len(SOIs) * 4)), dtype=np.float, order='F'))


#####################################
# CNCalls:
#
# Args:
#
#
#
# Returns
#
def CNCalls(sex2Clust, exons, countsNorm, clusts2Samps, clusts2Ctrls, priors, SOIs, plotDir, logOddsArray):
    pdfFile = os.path.join(plotDir, "ResCallsByCluster_" + str(len(SOIs)) + "samps.pdf")
    # create a matplotlib object and open a pdf
    PDF = matplotlib.backends.backend_pdf.PdfPages(pdfFile)
    
    # when discriminating between genders, 
    # importance of identifying autosomes and gonosomes exons index
    # to make calls on associated reference groups.
    if sex2Clust:
        gonoIndex, _ = clusterSamps.genderDiscrimination.getGenderInfos(exons)
        maskAutosome_Gonosome = ~np.isin(np.arange(countsNorm.shape[0]), sorted(set(sum(gonoIndex.values(), []))))

    ##############################
    # first loop 
    # Browse clusters
    for clustID in clusts2Samps:
        # recovery of data specific to the current cluster
        # sampleIndex2Process (list[int]): indexes of interest samples (from the cluster + controls)
        # exonsIndex2Process (list[int]): indexes of the exons having allowed the formation of the cluster
        (sampleIndex2Process, exonsIndex2Process) = extractClusterDependentDataPrivate(clustID, clusts2Samps, clusts2Ctrls, sex2Clust, maskAutosome_Gonosome)
        
        # Create Boolean masks for columns and rows
        col_mask = np.isin(np.arange(countsNorm[1]), sampleIndex2Process, invert=True)
        row_mask = np.isin(np.arange(countsNorm[0]), exonsIndex2Process, invert=True)

        # Use the masks to index the 2D numpy array
        clusterCounting = countsNorm[np.ix_(row_mask, col_mask)]

        ###########
        # Initialize  a hash allowing to detail the filtering carried out 
        # as well as the calls for all the exons. 
        # It is used for the pie chart representing the filtering.
        filterRes = dict.fromkeys(["med=0","cannotFitRG", "meanRG=0", "pseudoZscore<3", "sampleContribution2RG<0.5"], 0)

        ##################################
        # smoothing on the set of coverage data averaged by exons.
        # fit a gamma distribution on the non-captured exons.
        # - gammaParameters [list[float]]: [shape, loc, scale]
        # - gammaThreshold [float]: threshold delimiting exons not covered and exons covered 
        # (95% gamma cdf)
        gammaParameters, gammaThreshold = fitGammaDistributionPrivate(clusterCounting, clustID, PDF)

        ##############################
        # second loop 
        # Browse cluster-specific exons 
        for exon in range(clusterCounting.shape[0]):
             # Print progress every 10000 exons
            if exon % 10000 == 0:
                logger.info("ClusterID %s: %s  %s ", clustID, exon, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

            # Get count data for the exon
            exonFPM = clusterCounting[exon]

            robustGaussianParams = exonFilteringPrivate(exonFPM, gammaThreshold, filterRes)
            if robustGaussianParams is None:
                continue
            
            filterRes["ExonsCalls"]+=1
            
            ###################
            # 
            ###################
            # Retrieve results for each sample XXXXXX
            for i in clusts2Samps[clustID]:
                sample_data = exonFPM[sampleIndex2Process.index(i)]
                sampIndexInLogOddsArray = i * 4

                log_odds = computeLogOddsPrivate(sample_data, gammaParameters, gammaThreshold, priors, robustGaussianParams)

                for val in range(4):
                    if logOddsArray[exonsIndex2Process[exon], (sampIndexInLogOddsArray + val)] == 0:
                        logOddsArray[exonsIndex2Process[exon], (sampIndexInLogOddsArray + val)] = log_odds[val]
                    else:
                        logger.error('erase previous logOdds value')

        filtersPiePlotPrivate(clustID, filterRes, PDF)

    # close the open pdf
    PDF.close()
    return(logOddsArray)



###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
# extractClusterDependentDataPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# extraction of indexes specific to the samples contained in the cluster and 
# indexes of the exons to be processed (specific to autosomes or gonosomes)
# Args:
# - clustID [str] : cluster identifier
# - clusts2Samps (dict[str: list[int]]): for each cluster identifier a list of SOIs is associated
# - clusts2Ctrls (dict[str: list[str]]): for each target cluster identifier a list of control clusters is associated
# - sex2Clust (dict[str, list[str]]): for "A" autosomes or "G" gonosomes a list of corresponding clusterIDs is associated
# - mask (numpy.ndarray[bool]): boolean mask 1: autosome exon indexes, 0: gonosome exon indexes. dim=NbExons
# Returns a tupple (), each object are created here:
# - sampleIndex2Process (list[int]): SOIs indexes in current cluster
# - exonsIndex2Process (list[int]): exons indexes to treat the current cluster
def extractClusterDependentDataPrivate(clustID, clusts2Samps, clusts2Ctrls, sex2Clust=None, mask=None):
    ##################################
    ## Select cluster specific indexes to apply to countsNorm
    ##### COLUMN indexes:
    # Get the indexes of the samples in the cluster and its controls
    sampleIndex2Process = clusts2Samps[clustID]
    if clustID in clusts2Ctrls:
        for controls in clusts2Ctrls[clustID]:
            sampleIndex2Process.extend(clusts2Samps[controls])
    
    ##### ROW indexes:
    # in case there are specific autosome and gonosome clusters.
    # identification of the indexes of the exons associated with the gonosomes or autosomes.
    if sex2Clust:
        if clustID in sex2Clust["A"]:
            exonsIndex2Process = np.flatnonzero(mask)
        else:
            exonsIndex2Process = np.where(~mask)[0]
    else:
        exonsIndex2Process = range(len(mask))

    return(sampleIndex2Process, exonsIndex2Process)


############################
# fitGammaDistributionPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Estimate the parameters of the gamma distribution that best fits the data.
# The gamma distribution was chosen after testing 101 continuous distribution laws,
#  -it has few parameters (3 in total: shape, loc, scale=1/beta),
#  -is well-known, and had the best goodness of fit on the empirical data.
# Arg:
# - clusterCounting (np.ndarray[floats]): normalised fragment count table for individuals in a cluster
# - clustID (str): cluster identifier
# - PDF (matplotlib object): store plots in a single pdf
# Returns a tupple (gamma_parameters, threshold_value), each variable is created here:
# - gammaParams (tuple of floats): estimated parameters of the gamma distribution
# - uncovExonThreshold (float): value corresponding to 95% of the cumulative distribution function
# from the gamma, corresponds to the FPM threshold where before this the exons are not covered 
# (contains both uncaptured, poorly covered and potentially homodeleted exons).
def fitGammaDistributionPrivate(clusterCounting, clustID, PDF):
    # compute meanFPM by exons
    # save computation time instead of taking the raw data (especially for clusters with many samples)
    meanCountByExons = np.mean(clusterCounting, axis=1)

    # smooth the coverage profile with kernel-density estimate using Gaussian kernels
    # - binEdges (np.ndarray[floats]): FPM range from 0 to 10 every 0.1
    # - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
    #   dim= len(binEdges)
    binEdges, densityOnFPMRange = clusterSamps.smoothing.smoothingCoverageProfile(meanCountByExons)

    # recover the threshold of the minimum density means before an increase
    # - minIndex (int): index from "densityMeans" associated with the first lowest
    # observed mean
    (minIndex, _) = clusterSamps.smoothing.findLocalMin(densityOnFPMRange)

    countsExonsNotCovered = meanCountByExons[meanCountByExons <= binEdges[minIndex]]

    countsExonsNotCovered.sort()  # sort data in-place

    # estimate the parameters of the gamma distribution that best fits the data
    gammaParams = scipy.stats.gamma.fit(countsExonsNotCovered)

    # compute the cumulative distribution function of the gamma distribution
    cdf = scipy.stats.gamma.cdf(countsExonsNotCovered, a=gammaParams[0], loc=gammaParams[1], scale=gammaParams[2])

    # find the index of the last element where cdf < 0.95
    thresholdIndex = np.where(cdf < 0.95)[0][-1]

    # compute the value corresponding to 95% of the cumulative distribution function
    # this value corresponds to the FPM value allowing to split covered exons from uncovered exons
    uncovExonThreshold  = countsExonsNotCovered[thresholdIndex]

    coverageProfilPlotPrivate(clustID, binEdges, densityOnFPMRange, minIndex, uncovExonThreshold, clusterCounting.shape[1], PDF)

    return (gammaParams, uncovExonThreshold)

###################################
# coverageProfilPlotPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# generates a plot per cluster
# x-axis: the range of FPM bins (every 0.1 between 0 and 10)
# y-axis: exons densities
# black curve: density data smoothed with kernel-density estimate using Gaussian kernels
# red vertical line: minimum FPM threshold, all uncovered exons are below this threshold
# green curve: gamma fit
#
# Args:
# - sampleName (str): sample exact name
# - binEdges (np.ndarray[floats]): FPM range
# - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
#   dim= len(binEdges)
# - minIndex (int): index associated with the first lowest density observed
# - uncovExonThreshold (float): value corresponding to 95% of the cumulative distribution function
# from the gamma, corresponds to the FPM threshold where before this the exons are not covered 
# (contains both uncaptured, poorly covered and potentially homodeleted exons).
# - SOIsNb (int): number of samples in the cluster
# - pdf (matplotlib object): store plots in a single pdf
# Returns and saves a plot in the output pdf
def coverageProfilPlotPrivate(clustID, binEdges, densityOnFPMRange, minIndex, uncovExonThreshold, SOIsNb, PDF):

    fig = matplotlib.pyplot.figure(figsize=(6,6))
    matplotlib.pyplot.plot(binEdges, densityOnFPMRange, color='black', label='smoothed densities')
    matplotlib.pyplot.axvline(binEdges[minIndex], color='crimson', linestyle='dashdot', linewidth=2,
                label="minFPM=" + '{:0.1f}'.format(binEdges[minIndex]))
    matplotlib.pyplot.axvline(uncovExonThreshold, color='blue', linestyle='dashdot', linewidth=2,
                label="uncovExonThreshold=" + '{:0.2f}'.format(uncovExonThreshold))
    matplotlib.pyplot.ylim(0, 0.5)
    matplotlib.pyplot.ylabel("Exon densities")
    matplotlib.pyplot.xlabel("Fragments Per Million")
    matplotlib.pyplot.title("ClusterID:" + clustID + " coverage profile (" + str(SOIsNb) + ")")
    matplotlib.pyplot.legend()

    PDF.savefig(fig)
    matplotlib.pyplot.close()

    
###############################################################
# 
def exonFilteringPrivate (exonFPM, gammaThreshold, filterRes):
    ###################
    # Filter n°1: exon not covered 
    # treats several possible cases:
    # - all samples in the cluster haven't coverage for the current exon
    # - more than 2/3 of the samples have no cover.
    #   Warning: Potential presence of homodeletions. We have chosen don't
    # call them because they affect too many samples
    # exon is not kept for the rest of the filtering and calling step
    medianFPM = np.median(exonFPM)
    if medianFPM  == 0:
        filterRes["med=0"]+=1
        return

    ###################
    # fits a Gaussian robustly from the exon count data
    # meanRG [float] and stdevRG [float] are the gaussian parameters
    # Filter n°2: the Gaussian fitting cannot be performed 
    # the median (consider as the mean parameter of the Gaussian) is located
    # in an area without point data.
    # in this case exon is not kept for the rest of the filtering and calling step
    try:
        meanRG, stdevRG = fitRobustGaussianPrivate(exonFPM)
    except Exception as e:
        if str(e) == "cannot fit":
            filterRes["cannotFitRG"]+=1
            return
        else:
            raise
        
    ###################
    # Filter n°3: 
    # principle: the Gaussian obtained in a robust way must not be associated 
    # with a copie number total loss (CN0) 
    # a pseudozscore allows to exclude exons with a Gaussian overlapping the 
    # threshold of not covered exons (gammaThreshold). 
    # To obtain the pseudoZscore it's necessary that the parameters of the 
    # robust Gaussian != 0.
    
    # exon is not kept for the rest of the filtering and calling step
    if meanRG == 0:
        filterRes["meanRG=0"]+=1
        return
    
    # the mean != zero and all samples have the same coverage value.
    # In this case a new arbitrary standard deviation is calculated 
    # (simulates 5% on each side of the mean)
    if (stdevRG == 0) :
        stdevRG = meanRG / 20
        
    z_score = (meanRG - gammaThreshold) / stdevRG
    
    # the exon is excluded if there are less than 3 standard deviations between 
    # the threshold and the mean.
    if (z_score < 3):
        filterRes["pseudoZscore<3"]+=1
        return

    ###################
    # Filter n°4:
    # principle: Calls are considered possible when the robust Gaussian has a sample
    # minimum contribution of 50%.
    # otherwise exon is not kept for the calling step
    weight = computeWeightPrivate(exonFPM, meanRG, stdevRG)
    if (weight < 0.5):
        filterRes["sampleContribution2RG<0.5"]+=1
        return       
    
    return(meanRG, stdevRG)
    
###################################
# robustGaussianFitPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Fits a single principal gaussian component around a starting guess point
# in a 1-dimensional gaussian mixture of unknown components with EM algorithm
# script found to :https://github.com/hmiemad/robust_Gaussian_fit (v01_2023)
# Args:
# - X (np.array): A sample of 1-dimensional mixture of gaussian random variables
# - mu (float, optional): Expectation. Defaults to None.
# - sigma (float, optional): Standard deviation. Defaults to None.
# - bandwidth (float, optional): Hyperparameter of truncation. Defaults to 2.
# - eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.
# Returns:
# - mu [float],sigma [float]: mean and stdev of the gaussian component
def fitRobustGaussianPrivate(X, mu = None, sigma = None, bandwidth = 2.0, eps = 1.0e-5):
    if mu is None:
        #median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)
    mu_0 = mu + 1
    
    if sigma is None:
        #rule of thumb
        sigma = np.std(X)/3
    sigma_0 = sigma + 1
    
    bandwidth_truncated_normal_sigma = truncated_integral_and_sigma(bandwidth)

    while abs(mu - mu_0) + abs(sigma - sigma_0) > eps:
        #loop until tolerence is reached
        """
        create a uniform window on X around mu of width 2*bandwidth*sigma
        find the mean of that window to shift the window to most expected local value
        measure the standard deviation of the window and divide by the standard deviation of a truncated gaussian distribution
        measure the proportion of points inside the window, divide by the weight of a truncated gaussian distribution
        """
        Window = np.logical_and(X - mu - bandwidth * sigma < 0 , X - mu + bandwidth * sigma > 0)
        
        # condition to identify exons with points arround at the median 
        if Window.any():
            mu_0, mu = mu, np.average(X[Window])
            var = np.average(np.square(X[Window])) - mu**2
            sigma_0 , sigma = sigma, np.sqrt(var)/bandwidth_truncated_normal_sigma
        # no points arround the median
        # e.g. exon where more than 1/2 of the samples have an FPM = 0. 
        # A Gaussian fit is impossible => raise exception
        else:
            raise Exception("cannot fit") 
    return (mu, sigma)

###########
# normal_erf
# ancillary function of the robustGaussianFitPrivate function 
# computes Gauss error function
# The error function (erf) is used to describe the Gaussian or Normal distribution. 
# It gives the probability that a random variable follows a given Gaussian distribution, 
# indicating the probability that it is less than or equal to a given value. 
# In other words, the error function quantifies the probability distribution for a 
# random variable following a Gaussian distribution.
# this function replaces the use of the scipy.stats.erf module
def normal_erf(x, mu = 0, sigma = 1,  depth = 50):
    ele = 1.0
    normal = 1.0
    x = (x - mu)/sigma
    erf = x
    for i in range(1,depth):
        ele = - ele * x * x/2.0/i
        normal = normal + ele
        erf = erf + ele * x / (2.0 * i + 1)

    return np.clip(normal/np.sqrt(2.0*np.pi)/sigma,0,None) , np.clip(erf/np.sqrt(2.0*np.pi)/sigma,-0.5,0.5)

##########
# truncated_integral_and_sigma
# ancillary function of the robustGaussianFitPrivate function 
# allows for a more precise and focused analysis of a function 
# by limiting the study to particular parts of its defining set.
def truncated_integral_and_sigma(x):
    n,e = normal_erf(x)
    return np.sqrt(1-n*x/e)


############################
# computeWeightPrivate[PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# compute the sample contribution to the Gaussian obtained in a robust way.
#
# Args:
# - fpm_in_exon (np.ndarray[float]): FPM values for a particular exon for each sample
# - mean (float): mean FPM value for the exon
# - standard_deviation (float): std FPM value for the exon
# Returns weight of sample contribution to the gaussian for the exon [float]
@numba.njit
def computeWeightPrivate(fpm_in_exon, mean, standard_deviation):
    targetData = fpm_in_exon[(fpm_in_exon > (mean - (2 * standard_deviation))) &
                             (fpm_in_exon < (mean + (2 * standard_deviation))), ]
    weight = len(targetData) / len(fpm_in_exon)

    return weight


############################
# computeLogOddsPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Given four models, the log odds ratio (LOR) allows to choose the best-fitting model.
# Use of Bayes' theorem to deduce it.
#
# Args:
# - sample_data (float): a sample data point, FPM value
# - gamma_params (list(float)): estimated parameters of the gamma distribution [shape, loc, scale]
# - gamma_threshold (float):
# - prior_probabilities (list[float]): prior probabilities for different cases
# - mean (float): mean value for the normal distribution
# - standard_deviation (float): the standard deviation for the normal distribution
#
# Returns:
# - log_odds (list[float]): log-odds ratios for each copy number (CN0,CN1,CN2,CN3+)
def computeLogOddsPrivate(sample_data, params, gamma_threshold, prior_probabilities, mean, standard_deviation):
    # CN2 mean shift to get CN1 mean
    mean_cn1 = mean / 2

    # To Fill
    # Initialize an empty numpy array to store the probability densities for each copy number type
    probability_densities = np.zeros(4)

    ###############
    # Calculate the probability density for the gamma distribution (CN0 profil)
    # This is a special case because the gamma distribution has a heavy tail,
    # which means that the probability of density calculated from it can override
    # the other Gaussian distributions.
    # A condition is set up to directly associate a value of pdf to 0 if the sample FPM value
    # is higher than the mean of the Gaussian associated to CN1.
    # Reversely, the value of the pdf is truncated from the threshold value discriminating
    # covered from uncovered exons.
    cdf_cno_threshold = scipy.stats.gamma.cdf(gamma_threshold, a=params[0], loc=params[1], scale=params[2])
    if sample_data <= mean_cn1:
        probability_densities[0] = (1 / (1 - cdf_cno_threshold)) * scipy.stats.gamma.pdf(sample_data, a=params[0], loc=params[1], scale=params[2])

    ################
    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    probability_densities[1] = scipy.stats.norm.pdf(sample_data, mean / 2, standard_deviation)
    probability_densities[2] = scipy.stats.norm.pdf(sample_data, mean, standard_deviation)
    probability_densities[3] = scipy.stats.norm.pdf(sample_data, 3 * mean / 2, standard_deviation)

    #################
    # Add prior probabilities
    probability_densities_priors = np.multiply(probability_densities, prior_probabilities)

    ################
    # case where one of the probabilities is equal to 0 addition of an epsilon 
    # which is 1000 times lower than the lowest probability
    probability_densities_priors = addEpsilonPrivate(probability_densities_priors)

    ##################
    # Calculate the log-odds ratios
    emissionProba = np.zeros(4)
    for i in range(len(probability_densities_priors)):
        # Calculate the denominator for the log-odds ratio
        to_subtract = np.sum(probability_densities_priors[np.arange(probability_densities_priors.shape[0]) != i])

        # Calculate the log-odds ratio for the current probability density
        log_odd = np.log10(probability_densities_priors[i]) - np.log10(to_subtract)

        # probability transformation
        emissionProba[i]=1/(1+np.exp(log_odd))

    return emissionProba/emissionProba.sum() # normalized 

################
# addEpsilonPrivate
@numba.njit
def addEpsilonPrivate(probs, epsilon_factor=1000):
    min_prob = np.min(probs[probs > 0])
    epsilon = min_prob / epsilon_factor
    probs = np.where(probs == 0, epsilon, probs)
    return probs

###################################
# filtersPiePlotPrivate:
# generates a plot per cluster
# Args:
# - clustID [str]: cluster identifier 
# - filterRes (dict[str:int]): dictionary of exon counters of different filtering
# performed for the cluster
# - pdf (matplotlib object): store plots in a single pdf
#
# save a plot in the output pdf
def filtersPiePlotPrivate(clustID, filterRes, pdf):

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    matplotlib.pyplot.pie(filterRes.values(), labels=filterRes.keys(),
            colors=["grey", "yellow", "indianred", "mediumpurple", "royalblue", "mediumaquamarine"],
            autopct=lambda x: str(round(x, 2)) + '%',
            startangle=-270,
            pctdistance=0.7,
            labeldistance=1.1)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("filtered and called exons for the cluster " + clustID)

    pdf.savefig(fig)
    matplotlib.pyplot.close()
