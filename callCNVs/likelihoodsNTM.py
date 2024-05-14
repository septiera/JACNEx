import logging
import numpy
import scipy.stats
import statistics

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
# - likelihoods: numpy 3D-array of floats (pre-allocated) of size
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
# - if one of the fitCN2() QC criteria, exon is NOCALL => set likelihoods to -1;
# - else:
#   - rescale FPMS and CN2 model so new CN2 mean == 1 (idea: pdfs integrate to 1, so
#     without this rescaling high-FPM exons have small likelihoods compared to
#     low-FPM exons, resulting in smaller scores for CNVs that span them)
#   - calculate likelihoods for CN1, CN2, CN3+
#
# Args:
# - FPMsOfCluster: 2D-array of floats, FPMsOfCluster[e,s] is the FPM count for
#   exon e in sample s, the samples must include those in FITWITH clusters (in
#   addition to the samples in the current cluster of interest)
# - samplesOfInterest: 1D-array of bools, size = number of samples in FPMsOfCluster,
#   value==True if the sample is in the cluster of interest (vs being in a FITWITH cluster)
# - likelihoods: numpy 3D-array of floats (pre-allocated) of size
#   nbSamplesOfInterest * nbExons (==nbRows in FPMsOfCluster) * nbStates;
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
# - isHaploid: bool, if True this cluster of samples is assumed to be haploid
#   for all chromosomes where the exons are located (eg chrX and xhrY in men).
#
# Returns (CN2Means):
# - CN2means: 1D-array of nbExons floats, CN2means[e] is the fitted mean of
#   the CN2 model (before rescaling) of exon e for the cluster, or -1 if exon
#   is NOCALL
def fitCN2andCalcLikelihoods(FPMsOfCluster, samplesOfInterest, likelihoods, fpmThreshold, isHaploid):
    # sanity
    nbExons = FPMsOfCluster.shape[0]
    nbSamplesTotal = FPMsOfCluster.shape[1]
    nbSOIs = samplesOfInterest.sum()
    if ((nbExons != likelihoods.shape[1]) or
        (nbSamplesTotal != samplesOfInterest.shape[0]) or
        (nbSOIs != likelihoods.shape[0])):
        logger.error("sanity check failed in fitCN2andCalcLikelihoods(), impossible!")
        raise Exception("fitCN2andCalcLikelihoods sanity check failed")

    CN2means = numpy.full(nbExons, fill_value=-1, dtype=numpy.float64)

    if isHaploid:
        # set all likelihoods of CN1 to zero
        likelihoods[:, :, 1] = 0

    for ei in range(nbExons):
        (cn2Mu, cn2Sigma) = fitCN2(FPMsOfCluster[ei, :], fpmThreshold, isHaploid)
        if cn2Mu < 0:
            # exon is NOCALL for the whole cluster, squash likelihoods to -1
            likelihoods[:, ei, :] = -1.0
            # we could also calculate statistics on which QC criteria failed in
            # fitCN2(), based on the cn2Mu values, but no time now
            continue

        # save CN2 mean before rescaling
        CN2means[ei] = cn2Mu
        # now rescale CN2 sigma and FPMs of SOIs
        cn2SigmaRescaled = cn2Sigma / cn2Mu
        fpmsRescaled = FPMsOfCluster[ei, samplesOfInterest] / cn2Mu

        # CN1: shift the CN2 Gaussian so mean==0.5 (a single copy rather than 2)
        cn1Dist = statistics.NormalDist(mu=0.5, sigma=cn2SigmaRescaled)

        # CN2 model: the fitted Gaussian
        cn2Dist = statistics.NormalDist(mu=1.0, sigma=cn2SigmaRescaled)

        # CN3 model, as defined in cn3Dist()
        cn3Dist = cn3Distrib(1.0, cn2SigmaRescaled, isHaploid)

        for si in range(nbSOIs):
            if not isHaploid:
                likelihoods[si, ei, 1] = cn1Dist.pdf(fpmsRescaled[si])
            likelihoods[si, ei, 2] = cn2Dist.pdf(fpmsRescaled[si])
            likelihoods[si, ei, 3] = cn3Dist.pdf(fpmsRescaled[si])

    return(CN2means)


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
# - CN1 Gaussian (or CN2 Gaussian if isHaploid) can't be clearly distinguished
#   from CN0 model (see minZscore)
#
# If all QC criteria pass, return (mu,sigma) == mean and stdev of the CN2 model;
# otherwise return (E, 0) where E<0 depends on the first criteria that failed:
# E==-1 if exon isn't captured
# E==-2 if robustGaussianFit failed
# E==-3 if low support for CN2 model
# E==-4 if CN2 is too close to CN0
def fitCN2(FPMsOfExon, fpmThreshold, isHaploid):
    if numpy.median(FPMsOfExon) <= fpmThreshold:
        # uncaptured exon
        return(-1, 0)

    try:
        (mu, sigma) = callCNVs.robustGaussianFit.robustGaussianFit(FPMsOfExon)
    except Exception as e:
        if str(e) != "cannot fit":
            logger.warning("robustGaussianFit failed unexpectedly: %s", repr(e))
            raise
        else:
            return(-2, 0)

    # require at least minSamps samples within sdLim sigmas of mu
    minSamps = len(FPMsOfExon) * 0.5
    sdLim = 2
    samplesUnderCN2 = numpy.sum(numpy.logical_and(FPMsOfExon - mu - sdLim * sigma < 0,
                                                  FPMsOfExon - mu + sdLim * sigma > 0))
    if samplesUnderCN2 < minSamps:
        # low support for CN2
        return(-3, 0)

    # require CN1 (CN2 if haploid) to be at least minZscore sigmas from fpmThreshold
    minZscore = 3
    if ((not isHaploid and ((mu / 2 - minZscore * sigma) <= fpmThreshold)) or
        (isHaploid and ((mu - minZscore * sigma) <= fpmThreshold))):
        # CN1 / CN2 too close to CN0
        return(-4, 0)

    return(mu, sigma)


############################################
# cn3Distrib:
# build a statistical model of CN3+, based on the CN2 mu and sigma.
# CN3+ is modeled as a Gamma distribution (as implemented by AS) for now, although
# it's empirical and I'm not sure it behaves as expected after rescaling?
#
# Args:
# - (mu, sigma) of the CN2 model
# - isHaploid boolean (if isHaploid, CN3+ should look for 2x mu FPM rather than 1.5x)
#
# Return an object with a pdf() method - currently a "frozen" scipy distribution, but
# it could be some other object (eg statistics.NormalDist).
def cn3Distrib(mu, sigma, isHaploid):
    ### NOTES: planning to switch to a LogNormal distribution, AS is currently
    ### studying this - we need to decide how we set the LogNormal params
    ### ALSO we want to take isHaploid into account

    # The 'shape' parameter is hard-coded here, based on empirical testing.
    # A higher 'shape' concentrates the distribution around the mean and reduces the spread
    shape = 6
    # The 'loc' parameter shifts the gamma distribution to avoid significant overlap
    # with the Gaussian CN2 distribution
    loc = mu + sigma
    # The 'scale' parameter: the logarithm helps to reduce the scale of values, making the model more
    # adaptable and stable, especially if loc varies over a wide range.
    scale = numpy.log10(loc + 1)

    cn3dist = scipy.stats.gamma(shape, loc=loc, scale=scale)
    return(cn3dist)
