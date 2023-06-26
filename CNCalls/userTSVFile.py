import logging
import gzip
import numpy as np

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
# parseUserTSV
# Given a user-provided TSV file containing sample names and exon identifiers,
# along with a list of sample names and a list of exon definitions.
# It parses the TSV file to generate a list of exons to check for each sample.
# The function returns a list where each index corresponds to a sample and contains
# the indexes of matching exons from the exon definitions.
#
# Args:
# - userTSV (str): Path to the user TSV file.
# - samples (list[str]): List of sample names.
# - exons (list of lists[str, int, int, str]): List of exon definitions [CHR, START, END, EXONID].
#
# Returns:
# - samps2Check (list of lists[int]): List of exons to check for each sample.
#                                     Each index of "samples" corresponds to a list of matching "exons" indexes.
def parseUserTSV(userTSV, samples, exons):
    try:
        if userTSV.endswith(".gz"):
            userTSVFH = gzip.open(userTSV, "rt")
        else:
            userTSVFH = open(userTSV, "r")
    except Exception as e:
        logger.error("Opening provided userTSV %s: %s", userTSV, e)
        raise Exception('cannot open userTSV')

    samps2Check = [[] for _ in range(len(samples))]

    for line in userTSVFH:
        splitLine = line.rstrip().split("\t", maxsplit=1)
        # Check if the sample name is in the provided sample name list
        if splitLine[0] in samples:
            exonIDFound = False
            for exonID in range(len(exons)):
                # Check if the exon identifier is present in "exons"
                if exons[exonID][3].startswith(splitLine[1]):
                    samps2Check[samples.index(splitLine[0])].append(exonID)
                    exonIDFound = True

            if not exonIDFound:
                logger.error("Exon identifier %s has no match with the bed file provided in step 1.", splitLine[1])
                raise Exception("Exon identifier not found.")
        else:
            logger.error("Sample name %s has no match with the sample name list.", splitLine[0])
            raise Exception("Sample name not found.")

    userTSVFH.close()
    return samps2Check
