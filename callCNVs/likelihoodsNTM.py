import logging
import numpy
import scipy.stats

####### JACNEx modules
import callCNVs.robustGaussianFit


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
    return(numpy.full((nbSamples, nbExons, nbStates), fill_value=-1,
                      dtype=numpy.float64, order='F'))


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


############################################
# fitCN2andCalcLikelihoods:
# for each exon (==row of FPMsOfCluster):
# - fit a normal distribution to the dominant component of the FPMs (this is
#   our model of CN2, we assume that most samples are CN2)
# - if fitting fails one of the QC criteria, exon is NOCALL => set likelihoods to -1;
# - else:
#   - rescale FPMS so CN2 mean == 1 (goal: have comparable likelihoods between exons)
#   - calculate likelihoods for CN1, CN2, CN3+
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
def fitCN2andCalcLikelihoods(FPMsOfCluster):
    return


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

############################################
# fitCN2:
# Try to fit a normal distribution to the dominant component of FPMsOfExon
# (this is our model of CN2, we assume that most samples are CN2), and apply
# the following QC criteria, testing if:
# - exon isn't captured (median FPM <= fpmThreshold)
# - fitting fails (exon is very atypical, can't make any calls)
# - CN2 model isn't supported by 50% or more of the samples
# - CN2 Gaussian can't be clearly distinguished from CN0 model
#
# If one of the QC criteria fails, raise an exception;
# otherwise return (mu,sigma) == mean and stdev of the CN2 model
def fitCN2(FPMsOfExon, fpmThreshold):
    if numpy.median(FPMsOfExon) <= fpmThreshold:
        raise Exception("uncaptured exon")

    try:
        (mu, sigma) = callCNVs.robustGaussianFit.robustGaussianFit(FPMsOfExon)
    except Exception as e:
        if str(e) != "cannot fit":
            logger.warning("robustGaussianFit failed unexpectedly: %s", repr(e))
        raise

    # require at least minSamps samples within sdLim sigmas of mu
    minSamps = len(FPMsOfExon) * 0.5
    sdLim = 2
    samplesUnderCN2 = numpy.sum(numpy.logical_and(FPMsOfExon - mu - sdLim * sigma < 0,
                                                  FPMsOfExon - mu + sdLim * sigma > 0))
    if samplesUnderCN2 < minSamps:
        raise Exception("low support for CN2")

    # require CN2 to be at least minZscore sigmas from fpmThreshold
    minZscore = 3
    if (mu - minZscore * sigma) <= fpmThreshold:
        raise Exception("CN2 too close to CN0")

    return(mu, sigma)
