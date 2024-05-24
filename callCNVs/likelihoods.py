import logging
import math
import numpy
import scipy.stats

####### JACNEx modules
import callCNVs.robustGaussianFit
import figures.plots

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# allocateLikelihoods:
# Allocate, initialize to -1 and return a numpy 3D-array of floats of size
#   nbSamples * nbExons * nbStates: likelihoods[s,e,cn] will store the
#   likelihood of state cn for exon e in sample s
def allocateLikelihoods(nbSamples, nbExons, nbStates):
    return(numpy.full((nbSamples, nbExons, nbStates), fill_value=-1,
                      dtype=numpy.float64, order='C'))


############################################
# fitCNO:
# fit a half-normal distribution with mode=0 (ie loc=0 for scipy) to all FPMs
# in intergenicFPMs.
#
# Args:
# - intergenicFPMs numpy 2D-array of floats of size nbIntergenics * nbSamples
#   holding the FPM-normalized counts for intergenic pseudo-exons
#
# Returns (CN0scale, fpmCn0):
# - CN0scale is the scale parameter of the fitted half-normal distribution
# - fpmCn0 is the FPM threshold up to which data looks like it could very possibly
#   have been produced by the CN0 model (set to fracPPF of the inverse CDF == quantile
#   function). This will be used later for filtering NOCALL exons.
def fitCNO(intergenicFPMs):
    # fracPPF hard-coded here, should be fine and universal
    fracPPF = 0.95
    (hnormloc, hnormscale) = scipy.stats.halfnorm.fit(intergenicFPMs.ravel(), floc=0)
    fpmCn0 = scipy.stats.halfnorm.ppf(fracPPF, loc=0, scale=hnormscale)
    return (hnormscale, fpmCn0)


############################################
# calcLikelihoodsCN0:
# calculate the likelihood of state CN0 for each exon in each sampleOfInterest
# present in FPMs.
# CN0 is modeled as a half-normal distrib with mode=0 and scale=CN0scale.
# Results are stored in likelihoods.
#
# Args:
# - FPMs: numpy 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is
#   the FPM-normalized count for exon e in sample s
# - samplesOfInterest: 1D-array of bools of size nbSamples, value==True iff the sample
#   is in the cluster of interest (vs being in a FITWITH cluster)
# - likelihoods: numpy 3D-array of floats of size nbSamplesOfInterest * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likelihood of state cn for exon e in sample s, will be
#   filled in-place for CN0
# - CN0scale (float): scale param of the half-normal distribution that fits the
#   CN0 data, as returned by fitCN0
#
# Returns nothing, likelihoods is updated in-place.
def calcLikelihoodsCN0(FPMs, samplesOfInterest, likelihoods, CN0scale):
    # sanity:
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    nbSOIs = samplesOfInterest.sum()
    if ((nbExons != likelihoods.shape[1]) or (nbSamples != samplesOfInterest.shape[0]) or
        (nbSOIs != likelihoods.shape[0])):
        logger.error("sanity check failed in calcLikelihoodsCN0(), impossible!")
        raise Exception("calcLikelihoodsCN0 sanity check failed")

    soi = 0
    for si in range(nbSamples):
        if samplesOfInterest[si]:
            likelihoods[soi, :, 0] = scipy.stats.halfnorm.pdf(FPMs[:, si], scale=CN0scale)
            soi += 1


############################################
# fitCN2andCalcLikelihoods:
# for each exon (==row of FPMs):
# - fit a normal distribution to the dominant component of the FPMs (this is
#   our model of CN2, we assume that most samples are CN2)
# - if one of the fitCN2() QC criteria fails, exon is NOCALL => set likelihoods to -1;
# - else calculate and fill likelihoods for CN1, CN2, CN3+
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is the FPM count
#   for exon e in sample s; it includes counts for the samplesOfInterest, but also for
#   the samples in FITWITH clusters (these are used for fitting the CN2)
# - samplesOfInterest: 1D-array of bools of size nbSamples, value==True iff the sample
#   is in the cluster of interest (vs being in a FITWITH cluster)
# - likelihoods: numpy 3D-array of floats of size nbSamplesOfInterest * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likelihood of state cn for exon e in sample s, will be
#   filled in-place
# - fpmCn0: up to this FPM value, data "looks like it's from CN0"
# - exonsToPlot: Dict with key==exonIndex, value==list of lists[sampleIndex, sampleID] for
#   which we need to plot the FPMs and CN0-CN3+ models
# - plotDir: subdir where plots will be created (if any)
# - clusterID: string, for logging
# - isHaploid: bool, if True this cluster of samples is assumed to be haploid
#   for all chromosomes where the exons are located (eg chrX and chrY in men).
#
# Returns CN2means: 1D-array of nbExons floats, CN2means[e] is the fitted mean of
#   the CN2 model of exon e for the cluster, or -1 if exon is NOCALL
def fitCN2andCalcLikelihoods(FPMs, samplesOfInterest, likelihoods, fpmCn0,
                             exonsToPlot, plotDir, clusterID, isHaploid):
    # sanity
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    nbSOIs = samplesOfInterest.sum()
    if ((nbExons != likelihoods.shape[1]) or (nbSamples != samplesOfInterest.shape[0]) or
        (nbSOIs != likelihoods.shape[0])):
        logger.error("sanity check failed in fitCN2andCalcLikelihoods(), impossible!")
        raise Exception("fitCN2andCalcLikelihoods sanity check failed")

    CN2means = numpy.full(nbExons, fill_value=-1, dtype=numpy.float64)

    # exonStatus: count the number of exons that passed (exonStatus[0]) or failed
    # (exonStatus[1..4]) the fitCN2() QC criteria. This is just for logging.
    exonStatus = numpy.zeros(5, dtype=float)

    # create a view with all FPMs, then squash with the FPMs of SOIs if needed
    FPMsSOIs = FPMs
    if nbSamples != nbSOIs:
        FPMsSOIs = FPMs[:, samplesOfInterest]

    if isHaploid:
        # set all likelihoods of CN1 to zero
        likelihoods[:, :, 1] = 0

    for ei in range(nbExons):
        (status, cn2Mu, cn2Sigma) = fitCN2(FPMs[ei, :], fpmCn0, isHaploid)

        if status < 0:
            # exon is NOCALL for the whole cluster, squash likelihoods to -1
            likelihoods[:, ei, :] = -1.0
            exonStatus[-status] += 1

        else:
            CN2means[ei] = cn2Mu
            exonStatus[0] += 1

            # CN1: shift the CN2 Gaussian so mean==cn2Mu/2 (a single copy rather than 2)
            if not isHaploid:
                likelihoods[:, ei, 1] = gaussianPDF(FPMsSOIs[ei, :], cn2Mu / 2, cn2Sigma)
            # else keep CN1 likelihood at zero as set above

            # CN2 model: the fitted Gaussian
            likelihoods[:, ei, 2] = gaussianPDF(FPMsSOIs[ei, :], cn2Mu, cn2Sigma)
            # NOTE: I tried statistics.normalDist but it's super slow:
            # cn1Dist = statistics.NormalDist(mu=cn1Mu, sigma=cn2Sigma)

            # CN3 model, as defined in cn3PDF()
            likelihoods[:, ei, 3] = cn3PDF(FPMsSOIs[ei, :], cn2Mu, cn2Sigma, isHaploid)

        # in all cases, plot the exon if requested
        if ei in exonsToPlot:
            figures.plots.plotExon(exonsToPlot[ei], exonsToPlot[ei], FPMsSOIs[ei, :],
                                   cn2Mu, cn2Sigma, isHaploid, fpmCn0)

    # log exon statuses (as percentages)
    exonStatus *= (100 / exonStatus.sum())
    toPrint = "exon QC summary for cluster " + clusterID + ":\n\t"
    toPrint += "%.1f%% CALLED, " % exonStatus[0]
    toPrint += "%.1f%% NOT-CAPTURED, %.1f%% FIT-CN2-FAILED, " % (exonStatus[1], exonStatus[2])
    toPrint += "%.1f%% CN2-LOW-SUPPORT, %.1f%% CN0-TOO-CLOSE" % (exonStatus[3], exonStatus[4])
    logger.info("%s", toPrint)

    return(CN2means)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

############################################
# fitCN2:
# Try to fit a normal distribution to the dominant component of FPMsOfExon
# (this is our model of CN2, we assume that most samples are CN2), and apply
# the following QC criteria, testing if:
# - exon isn't captured (median FPM <= fpmCn0)
# - fitting fails (exon is very atypical, can't make any calls)
# - CN2 model isn't supported by 50% or more of the samples
# - CN1 Gaussian (or CN2 Gaussian if isHaploid) can't be clearly distinguished
#   from CN0 model (see minZscore)
#
# Return (E, mu, sigma): E is the status/error code, mu and sigma are the mean
# and stdev of the CN2 model (set to (0,0) if we couldn't fit CN2, ie E==-1 or -2);
# E==0  if all QC criteria pass
# E==-1 if exon isn't captured
# E==-2 if robustGaussianFit failed
# E==-3 if low support for CN2 model
# E==-4 if CN1 (CN2 if isHaploid) is too close to CN0
def fitCN2(FPMsOfExon, fpmCn0, isHaploid):
    if numpy.median(FPMsOfExon) <= fpmCn0:
        # uncaptured exon
        return(-1, 0, 0)

    try:
        (mu, sigma) = callCNVs.robustGaussianFit.robustGaussianFit(FPMsOfExon)
    except Exception as e:
        if str(e) != "cannot fit":
            logger.warning("robustGaussianFit failed unexpectedly: %s", repr(e))
            raise
        else:
            return(-2, 0, 0)

    # require at least minSamps samples within sdLim sigmas of mu
    minSamps = len(FPMsOfExon) * 0.5
    sdLim = 2
    samplesUnderCN2 = numpy.sum(numpy.logical_and(FPMsOfExon - mu - sdLim * sigma < 0,
                                                  FPMsOfExon - mu + sdLim * sigma > 0))
    if samplesUnderCN2 < minSamps:
        # low support for CN2
        return(-3, mu, sigma)

    # require CN1 (CN2 if haploid) to be at least minZscore sigmas from fpmCn0
    minZscore = 3
    if ((not isHaploid and ((mu / 2 - minZscore * sigma) <= fpmCn0)) or
        (isHaploid and ((mu - minZscore * sigma) <= fpmCn0))):
        # CN1 / CN2 too close to CN0
        return(-4, mu, sigma)

    return(0, mu, sigma)


############################################
# cn3Distrib:
# build a statistical model of CN3+, based on the CN2 mu and sigma.
# CN3+ is modeled as a LogNormal that aims to:
# - captures data around 1.5x the CN2 mean (2x if isHaploid) and beyond
# - avoids overlapping too much with the CN2
# The LogNormal is heavy-tailed, which is nice because we are modeling CN3+ not CN3.
#
# Args:
# - (cn2Mu, cn2Sigma) of the CN2 model
# - isHaploid boolean (NOTE: currently not used - TODO)
#
# Return an object with a pdf() method that accepts a 1D numpy.ndarray as input and
# returns a 1D array of the same size containing the PDF values - currently we
# use a "frozen" scipy distribution, but it could be some other object
def cn3Distrib(cn2Mu, cn2Sigma, isHaploid):
    # LogNormal parameters set empirically
    # scipy shape == sigma
    shape = 0.5
    # scipy scale == exp(mu), we want mu=ln(cn2Mu)
    scale = cn2Mu
    # scipy loc shifts the distrib, we want avoid overlapping too much with CN2
    loc = cn2Mu + 2 * cn2Sigma

    cn3dist = scipy.stats.lognorm(shape, loc=loc, scale=scale)
    return(cn3dist)


############################################
# Precompute sqrt(2*pi), used tons of times in gaussianPDF()
SQRT_2PI = math.sqrt(2 * math.pi)


############################################
# Calculate the likelihoods (==values of the PDF) of a Gaussian distribution
# of parameters mu and sigma, at x (1D numpy.ndarray of floats).
# Returns a 1D numpy.ndarray, same size as x
def gaussianPDF(x, mu, sigma):
    return numpy.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * SQRT_2PI)


############################################
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN3+, based on the CN2 mu and sigma, at x (1D numpy.ndarray of floats).
#
# CN3+ is currently modeled as a LogNormal that aims to:
# - capture data around 1.5x the CN2 mean (2x if isHaploid) and beyond;
# - avoid overlapping too much with the CN2.
# The LogNormal is heavy-tailed, which is nice because we are modeling CN3+ not CN3.
#
# Args:
# - x: 1D numpy.ndarray of the FPMs for which we want to calculate the likelihoods
# - (cn2Mu, cn2Sigma) of the CN2 model
# - isHaploid boolean (NOTE: currently not used - TODO)
#
# Returns a 1D numpy.ndarray, same size as x
def cn3PDF(x, cn2Mu, cn2Sigma, isHaploid):
    # LogNormal parameters set empirically
    sigma = 0.5
    mu = math.log(cn2Mu)
    # shift the whole distribution by "loc" to avoid overlapping too much with CN2
    loc = cn2Mu + 2 * cn2Sigma
    # mask values <= loc to avoid doing any computations on them, this also copies the data
    xm = numpy.ma.masked_less_equal(x, loc)
    xm -= loc
    # the formula for the pdf of a LogNormal is pretty simple (see wikipedia), but
    # the order of operations is important to avoid overflows/underflows. The following
    # works for us and is reasonably fast
    res = numpy.ma.log(xm)
    res = numpy.ma.exp(-(((res - mu) / sigma)**2 / 2) - res - numpy.ma.log(sigma * SQRT_2PI))
    return (res.filled(fill_value=0.0))
