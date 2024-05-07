import logging
import numpy
import scipy.stats


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# fitCNO:
# fit a half-normal distribution with mode=0 (ie loc=0 for scipy) to all FPMs
# in intergenicFPMs.
#
# Args:
# - intergenicFPMs numpy 2D-array of floats, size=len(intergenics)] x len(samples),
#   holding the FPM-normalized counts for intergenic pseudo-exons
#
# Returns (CN0scale, fpmThreshold):
# - CN0scale is the scale parameter of the fitted hnorm distribution
# - fpmThreshold is the FPM threshold up to which data looks like it could very possibly
#   have been produced by the CN0 model (set to fracPPF of the inverse CDF == quantile
#   function)
def fitCNO(intergenicFPMs):
    fracPPF = 0.95
    (hnormloc, hnormscale) = scipy.stats.halfnorm.fit(intergenicFPMs.ravel(), floc=0)
    fpmThreshold = scipy.stats.halfnorm.ppf(fracPPF, loc=0, scale=hnormscale)
    return (hnormscale, fpmThreshold)


############################################
# calcLikelihoodsCN0:
# calculate the likelihood of state CN0 for each exon + sample present in FPMs.
# CN0 is modeled as a half-normal distrib with mode=0 and scale=CN0scale.
# Results are stored in likelihoodsDict.
#
# Args:
# - FPMs: numpy 2D-array of floats, FPMs[e,s] is the FPM-normalized count for exon
#   e in sample s - the caller must know what exons are present and in what order
# - samples: list of sampleIDs present in FPMs, in the same order
# - likelihoodsDict: key==sampleID, value is a numpy 2D-array of floats (pre-allocated)
#   of size nbExons (same as rows in FPMs) * nbStates.
# - CN0scale (float): scale param of the half-normal distribution that fits the
#   CN0 data, as returned by fitCN0
#
# Returns nothing, the values of likelihoodsDict are updated in-place.
def calcLikelihoodsCN0(FPMs, samples, likelihoodsDict, CN0scale):
    for si in range(len(samples)):
        sampleID = samples[si]
        likelihoodsDict[sampleID][:, 0] = scipy.stats.halfnorm.pdf(FPMs[:, si], scale=CN0scale)


