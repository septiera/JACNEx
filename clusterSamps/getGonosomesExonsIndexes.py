import numpy as np
import logging

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###############################################################################
# getSexChrIndexes:
# - identifies the sex chromosomes (gonosomes) presents in the "countsFile"
# input data and extracts the associated exon indexes.
# Please note that in this script only X, Y, Z and W chromosomes can be found.
# This is more than enough to cover a large part of the living world
# => mammals, birds, fish, reptiles.
#
# Args:
#  - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
# Returns a boolean numpy.ndarray where the indices of the exons
# corresponding to the gonosomes are True and False for the autosomes.
def getSexChrIndexes(exons):
    # sex chromosome list
    gonoChromList = ["X", "Y", "W", "Z"]

    # cases where the chromosome terminology does not match the sex chromosome list.
    # adding chr in front of names
    if exons[0][0].startswith("chr"):
        gonoChromList = ["chr" + letter for letter in gonoChromList]

    # creation of the boolean numpy.ndarray of exonsNb length
    chrSexMask = np.array([exon[0] in gonoChromList for exon in exons])
    return chrSexMask
