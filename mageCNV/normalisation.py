import numpy as np
import numba  # make python faster
import logging

# prevent numba DEBUG messages filling the logs when we are in DEBUG loglevel
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################################################
# Fragment Per Million normalisation
# Comparisons between samples are then possible
# For each sample :
# FPM = (Number of fragments mapped to an exon * 1e6)/Total number of mapped fragments
# Arg:
#   -countsArray : an int numpy array of counts, Dim=NbExons*NbSOIs
# Return a float numpy array of normalised counts, Dim=NbExon*NbSOIs
@numba.njit
def FPMNormalisation(countsArray):
    # create an empty array to filled with the normalized counts
    countsNorm = np.zeros_like(countsArray, dtype=np.float32)
    for sampleCol in range(countsArray.shape[1]):
        SampleCountsSum = np.sum(countsArray[:, sampleCol])
        SampleCountNorm = (countsArray[:, sampleCol] * 1e6) / SampleCountsSum  # 1e6 is equivalent to 1x10^6
        countsNorm[:, sampleCol] = SampleCountNorm
    return(countsNorm)

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
