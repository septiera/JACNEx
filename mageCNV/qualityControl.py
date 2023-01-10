import numpy as np
import logging
import mageCNV.slidingWindow
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# SampsQC :
# evaluates the coverage profile of the samples and identifies the uncovered exons for all samples.
# Args:
#  - counts (np.ndarray[float]): normalised fragment counts. dim = NbExons x NbSOIs
#  - SOIs (list[str]): samples of interest names
#  - windowSize (int): number of bins in a window
#  - outputFile (optionnal str): full path to save the pdf
# Returns a tupple (validityStatus, exons2RMAllSamps), each variable is created here:
#  - validityStatus (np.array[int]): the validity status for each sample (1: valid, 0: invalid), dim = NbSOIs
#  - exons2RMAllSamps (list[int]): indexes of uncovered exons common for all samples with a valid status
def SampsQC(counts, SOIs, windowSize, outputFile=None):
    
    # To Fill:
    validityStatus = np.ones(len(SOIs), dtype=np.int)

    # Accumulator:
    exons2RM = []
    validSampsCounter = 0

    # To remove not use only for dev control
    listtest = []

    if outputFile:
        pdf = matplotlib.backends.backend_pdf.PdfPages(outputFile)
        
    for sampleIndex in range(len(SOIs)):
        # extract sample counts
        sampCounts = counts[:, sampleIndex]

        # densities calculation by a histogram with a sufficiently accurate range (e.g: 0.05)
        start = 0
        stop = max(sampCounts)
        binEdges = np.linspace(start, stop, int(stop * 100))  # !!! hardcode
        binEdges = np.around(binEdges,2)
        
        # np.histogram function takes in the data to be histogrammed and a number of bins
        # - densities (np.array[floats]): number of elements in the bin / (bin width * total number of elements)
        # - fpmBins (np.array[floats]): values at which each bin starts and ends
        densities, fpmBins = np.histogram(sampCounts, bins=binEdges, density=True)

        # smooth the coverage profile from a sliding window. 
        # - middleWindowIndexes (list[int]): the middle indices of each window 
        #    (in agreement with the fpmBins indexes)
        # - densityMeans (list[float]): mean density for each window covered
        (middleWindowIndexes, densityMeans) =  mageCNV.slidingWindow.smoothingCoverageProfile(densities, windowSize)

        # recover the threshold of the minimum density means before an increase
        # - minIndex (int): index associated with the first lowest observed mean
        #   (in agreement with the fpmBins indexes)
        # - minDensitySum (float): first lowest observed mean
        (minIndex, minDensityMean) =  mageCNV.slidingWindow.findLocalMin(middleWindowIndexes, densityMeans)
        
        # recover the threshold of the maximum density means after the 
        # minimum density means which is associated with the largest covered exons number.
        # - maxIndex (int): index associated with the maximum density mean observed 
        #   (in agreement with the fpmBins indexes)
        # - maxDensity (float): maximum density mean
        (maxIndex, maxDensityMean) = findLocalMaxPrivate(middleWindowIndexes, densityMeans, minDensityMean)

        # graphic representation of coverage profiles.
        # returns a pdf in the output folder
        if outputFile:
            coverageProfilPlotPrivate(SOIs[sampleIndex], densities, middleWindowIndexes, densityMeans, minIndex, maxIndex, pdf)
            
            
        listtest.append([SOIs[sampleIndex], minIndex, minDensityMean, maxIndex, maxDensityMean])

        # sample removed where the minimum sum (corresponding to the limit of the poorly covered exons)
        # is less than 20% different from the maximum sum (corresponding to the profile of the most covered exons).
        if (((maxDensityMean - minDensityMean) / maxDensityMean) < 0.20):
            logger.warning("Sample %s has a coverage profile doesn't distinguish between covered and uncovered exons.", 
                           SOIs[sampleIndex])
            validityStatus[sampleIndex] = 0
        # If not, then the indices of exons below the minimum threshold are kept and only those common
        # to the previous sample analysed are kept.
        else:
            validSampsCounter += 1
            exons2RMSamp = np.where(sampCounts <= fpmBins[minIndex])
            if (len(exons2RM) != 0):
                exons2RM = np.intersect1d(exons2RM, exons2RMSamp)
            else:
                exons2RM = exons2RMSamp
                
    pdf.close()
    logger.info("Uncovered exons number %s to be deleted before clustering for %s/%s valid samples.", 
                len(exons2RM), validSampsCounter, len(SOIs))
    return(validityStatus, exons2RM, listtest)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###################################
# findLocalMaxPrivate:
# recover the threshold of the maximum density means after the 
# minimum density means which is associated with the largest covered exons number.
# Args:
#  - middleWindowIndexes (list[int]): the middle indices of each window 
#  - densityMeans (list[float]): mean density for each window covered
# this arguments are from the smoothingCoverProfile function.
#  - minIndex (int): index associated with the first lowest observed mean 
# this argument is from the findLocalMin function.
# Returns a tupple (minIndex, minDensity), each variable is created here:
#  - maxIndex (int): index associated with the maximum density mean observed 
#  - maxDensity (float): maximum density mean 
def findLocalMaxPrivate(middleWindowIndexes, densityMeans, minDensityMean):
    minMeanIndex=densityMeans.index(minDensityMean)
    maxDensityMean = max(densityMeans[minMeanIndex:])
    maxIndex = middleWindowIndexes[minMeanIndex + densityMeans[minMeanIndex:].index(maxDensityMean)]
    return (maxIndex, maxDensityMean)

####################################
# coverageProfilPlotPrivate:
# 
# Args:
# - densities (list of floats): list of coverage densities
# - fpmBins (list of floats): list of corresponding bin values
# - middleWindowIndexes (list[int]): the middle indices of each window 
# - densityMeans (list[float]): mean density for each window covered
# - minIndex (int): index associated with the first lowest observed mean 
# - maxIndex (int): index associated with the maximum density mean observed 
# - pdf 
# Returns a pdf file in the output folder
def coverageProfilPlotPrivate(sampleName, densities, middleWindowIndexes, densityMeans, minIndex, maxIndex, pdf):
    # Disable interactive mode
    plt.ioff()
    ##### Raw Profil
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(densities,'r')
    ax1.set_xlim(0,1000)
    ax1.set_ylim(0,1)
    ax1.set_ylabel("Density")
    ax1.set_xlabel("Fragment Per Million for each exons")
    ax1.set_title(sampleName+" raw coverage profile" )

    ##### Smooth Profil
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(middleWindowIndexes, densityMeans)
    ax2.set_xlabel("Fragment Per Million for each exons (Middle window index)")
    ax2.set_ylabel("Mean density by window")
    ax2.axvline(minIndex, color='crimson', linestyle='dashdot', linewidth=2, 
                 label=str(minIndex))
    ax2.axvline(maxIndex, color='darkorange', linestyle='dashdot', linewidth=2, 
                 label=str(maxIndex))
    ax2.set_xlim(0,1000)
    ax2.set_title(sampleName+" smoothing coverage profile")
    ax2.legend()
    
    plt.subplots_adjust(wspace=0.5)
    pdf.savefig(fig)
    plt.close()

