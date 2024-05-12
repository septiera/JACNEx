import logging
import numpy
import scipy.stats


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# allocateLikelihoods:
# Allocate, initialize to -1 and return a numpy 3D-array of floats of size
#   nbSamples * nbExons * nbStates: likelihoods[s,e,cn] will store the likehood
#   of state cn for exon e in sample s
def allocateLikelihoods(nbSamples, nbExons, nbStates):
    return(numpy.full((nbSamples, nbExons, nbStates), fill_value=-1, dtype=numpy.float64, order='F'))


############################################
# fitCNO:
# fit a half-normal distribution with mode=0 (ie loc=0 for scipy) to all FPMs
# in intergenicFPMs.
#
# Args:
# - intergenicFPMs numpy 2D-array of floats, size=len(intergenics)] * len(samples),
#   holding the FPM-normalized counts for intergenic pseudo-exons
#
# Returns (CN0scale, fpmThreshold):
# - CN0scale is the scale parameter of the fitted half-normal distribution
# - fpmThreshold is the FPM threshold up to which data looks like it could very possibly
#   have been produced by the CN0 model (set to fracPPF of the inverse CDF == quantile
#   function). This will be used later for filtering NOCALL exons.
def fitCNO(intergenicFPMs):
    fracPPF = 0.95
    (hnormloc, hnormscale) = scipy.stats.halfnorm.fit(intergenicFPMs.ravel(), floc=0)
    fpmThreshold = scipy.stats.halfnorm.ppf(fracPPF, loc=0, scale=hnormscale)
    return (hnormscale, fpmThreshold)


############################################
# calcLikelihoodsCN0:
# calculate the likelihood of state CN0 for each exon + sample present in FPMs.
# CN0 is modeled as a half-normal distrib with mode=0 and scale=CN0scale.
# Results are stored in likelihoods.
#
# Args:
# - FPMs: numpy 2D-array of floats, FPMs[e,s] is the FPM-normalized count for exon
#   e in sample s - the caller must know what samples and exons are present and in
#   what order
# - likelihoods:numpy 3D-array of floats (pre-allocated) of size
#   nbSamples (==nbColumns in FPMs) * nbExons (==nbRows in FPMs) * nbStates;
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
#   (same s and e indexes as in FPMs)
# - CN0scale (float): scale param of the half-normal distribution that fits the
#   CN0 data, as returned by fitCN0
#
# Returns nothing, likelihoods is updated in-place.
def calcLikelihoodsCN0(FPMs, likelihoods, CN0scale):
    for si in range(FPMs.shape[1]):
        likelihoods[si, :, 0] = scipy.stats.halfnorm.pdf(FPMs[:, si], scale=CN0scale)

