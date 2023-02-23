import logging
import gzip
import numpy as np
import numba


# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################
# printCNCallsFile:
# Args:
# - emissionArray (np.ndarray[float]): contain emission probabilities. dim=NbExons* (NbSOIs*[CN0,CN1,CN2,CN3+])
# - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# - SOIs (list[str]): sampleIDs copied from countsFile's header
# - outFile is a filename that doesn't exist, it can have a path component (which must exist),
#     output will be gzipped if outFile ends with '.gz'
#
# Print this data to outFile as a 'CNCallsFile'
def printCNCallsFile(emissionArray, exons, SOIs, outFile):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open CNCallsFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open CNCallsFile')

    toPrint = "CHR\tSTART\tEND\tEXON_ID\t"
    for i in SOIs:
        for j in range(4):
            toPrint += f"{i}_CN{j}_prob" + "\t"

    outFH.write(toPrint.rstrip())
    for i in range(len(exons)):
        toPrint = exons[i][0] + "\t" + str(exons[i][1]) + "\t" + str(exons[i][2]) + "\t" + exons[i][3]
        toPrint += calls2str(emissionArray, i)
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

##############################################################
# allocateProbsArray:
# Args:
# - numExons, numSamples
# Returns an float array with -1, adapted for storing the probabilities for each
# type of copy number. dim= NbExons x (NbSOIs x [CN0, CN1, CN2,CN3+])
def allocateProbsArray(numExons, numSamples):
    # order=F should improve performance
    return np.full((numExons, (numSamples * 4)), -1, dtype=np.float16, order='F')

#############################################################
# calls2str:
# return a string holding the calls from emissionArray[exonIndex],
# tab-separated and starting with a tab
@numba.njit
def calls2str(emissionArray, exonIndex):
    toPrint = ""
    for i in range(emissionArray.shape[1]):
        toPrint += "\t" + "{:0.2f}".format(emissionArray[exonIndex, i])
    return(toPrint)
