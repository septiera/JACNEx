import logging
# set up logger, using inherited config
logger = logging.getLogger(__name__)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# selectMinThreshold :
# Compute the low coverage threshold , the low density sum and the density sums list
# using a sliding window approach.  
# Args:
#  - densities (list[floats]): coverage densities
#  - windowSize (int): number of bins in a window 
# Returns a tupple (validityStatus, exons2RMAllSamps), each variable is created here:
#  - minIndex (int): low coverage threshold asssociated index (applicable for densities and fpmbins)
#  - minSum (float): low coverage threshold associated sum
#  - sums (list[floats]): complete list of sums calculated for all the windows covered
def selectMinThreshold(densities, windowSize):
    # Accumulators:
    minIndex = 0
    minSum = float("inf")    
    sums = []
    
    # compute the first window sum to facilitate the calculation in the loop
    rollingSum = sum(densities[:windowSize])
    sums.append(rollingSum)
    
    # compute and store all window sums
    for i in range(1, len(densities) - windowSize + 1):
        rollingSum += densities[i+windowSize-1] - densities[i-1]
        sums.append(rollingSum)
    
    # Find index of first minimum sum before an increase
    for i in range(len(sums)-1):
        # Check that minimum sum is the smallest sum seen after 20 windows
        minSum = sums[i]
        for j in range(i+1, i+20):
            if sums[j] < minSum:
                minSum = sums[j]
        
        # Break if minimum sum is still the smallest after 20 windows
        if sums[i] < sums[i+1] and minSum == sums[i]:
            minIndex = i + windowSize //2
            break
        
    return (minIndex, minSum, sums) 