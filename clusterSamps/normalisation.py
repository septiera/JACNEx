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
# Fragment counts are normalised in Fragment Per Million (FPM).
# algorithm for one sample :
# FPM = (Exon FragsNb * 1x10^6) / (Total FragsNb)
# This normalisation allows to compare samples with each other.
# It's therefore necessary before any processing on the data (e.g. clustering and calling)
# This small function enables numba optimizations.
# Arg:
# - countsArray (np.ndarray[int]): fragment counts, Dim=NbExons*NbSOIs
# Returns:
# - countsNorm (np.ndarray[float]): normalised counts of countsArray same dimension
# for arrays in input/output: NbExons*NbSOIs
@numba.njit
def FPMNormalisation(countsArray):
    # create an empty array to filled with the normalized counts
    countsNorm = np.zeros_like(countsArray, dtype=np.float32)
    for sampleCol in range(countsArray.shape[1]):
        SampleCountsSum = np.sum(countsArray[:, sampleCol])
        SampleCountNorm = (countsArray[:, sampleCol] * 1e6) / SampleCountsSum  # 1e6 is equivalent to 1x10^6
        countsNorm[:, sampleCol] = SampleCountNorm
    return countsNorm

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
