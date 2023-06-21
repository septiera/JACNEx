import logging
import numba
import numpy as np

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
# parseUserTSV
# Given a TSV file provided by the user with two columns [SAMPLES, EXONID],
# a list of sample names, and a list of lists representing exon definitions:
#
# Create a list of lists of length nbSamples.
# For each sample index list, append the "exons" indexes of the exons definition 
def parseUserTSV(userTSV, samples, exons):
