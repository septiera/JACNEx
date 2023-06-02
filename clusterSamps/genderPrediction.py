import numpy as np
import logging

# import sklearn submodule for Kmeans calculation
import sklearn.cluster

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###############################################################################
# exonOnSexChr: identify exons located on a sex chromosome, ie currently one of
# (X, Y, Z, W). This covers most species including mammals, birds, fish, reptiles.
#
# Arg:
#  - a list of exons, each exon is a list of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID
#
# Returns an uint8 numpy.ndarray of the same size as exons, value is:
# - 0 if the exon is on an autosome;
# - 1 if the exon is on the X or Z chromosome;
# - 2 if the exon is on the Y or W chromosome.
#
# Note that X or Z is present in one or two copies in each individual, and is
# (usually?) the larger of the two sex chromosomes; while Y or W is present
# in 0 or 1 copy and is smaller.
# However interpretation of "having two copies of X|Z" differs: in XY species
# (eg humans) XX is the Female, while in ZW species (eg birds) ZZ is the Male.
def exonOnSexChr(exons):
    # accepted sex chromosomes as keys, value==1 for X|Z and 2 for Y|W
    sexChroms = {"X": 1, "Y": 2, "W": 2, "Z": 1}
    # also accept the same chroms prepended with 'chr'
    for sc in list(sexChroms.keys()):
        sexChroms["chr" + sc] = sexChroms[sc]

    exonOnSexChr = np.zeros(len(exons), dtype=np.uint8)
    for ei in range(len(exons)):
        if exons[ei][0] in sexChroms:
            exonOnSexChr[ei] = sexChroms[exons[ei][0]]
    return(exonOnSexChr)


###############################################################################
# Assign a gender for each sample (ie column of exonsFPM)
#
# Args:
# - exonsFPM: np.ndarray of FPM-normalized counts (floats), size = nbOfExons x nbOfSamples
# - exonOnSexChr: uint8 np.ndarray of size = nbOfExons, as returned by exonOnSexChr()
#
# Returns an uint8 np.ndarray, size = nbOfSamples, value at index i is:
# - 1 if the sample/column index i is predicted to be Female
# - 2 if it's predicted to be Male
#
# If we're unable to assign a gender to each sample, log a warning and return an
# all-female assignment vector. This can happen legitimately if the cohort is single-gender,
# but it could also result from noisy / heterogeneous data, or flaws in our methodology.
def assignGender(exonsFPM, exonOnSexChr):
    sample2gender = np.ones(exonsFPM.shape[1], dtype=np.uint8)

    # TODO
    # idea: ignore Y (too small), just calculate the sum of FPMs of exons on chrX for
    # each sample (or Z but it doesn't change anything), ie exonOnSexChr == 1,
    # and hopyfully the result wll be easily interpretable, with 2 separate gaussians.
    # If needed this could be improved by only summing the FPMs for exons that are
    # captured in every sample (ie exonsFPM must be >= a small cutoff in every sample).
    # We could also do a K-means instead of looking for Gaussians.

    if predictGenderFailed:
        logger.warning("gender assignment failed. This is fine for single-gender cohorts,")
        logger.warning("but otherwise expect low-confidence CNV calls on gonosomes.")
        logger.warning("If your cohort is mixed-gender, please let us know so we can fix it.")
        logger.warning("Proceeding anyways, assuming arbitrarily that all samples are Female")

    return(sample2gender)


###############################################################################
# sexAssignment
# from gonosome exon definitions identify the species concerned: mammals or others
# (birds, reptiles, amphibians, fish)
# Performs a kmeans on the gonosome exon counts, to identify two groups associated
# with the sexe.
# Assigns sexe to each kmean groups based on their coverage ratios.
#
# Args:
# - gonosomesFPM (np.ndarray[float]): normalized fragment counts
# - gonosomesExons (list of list[str]): exon definitions
# - samples (list[str]): sample names
# Returns a list of lists[str] for each sexe is assigned the Kmeans samples list.
def sexAssignment(gonosomesFPM, gonosomesExons, samples):
    # To Fill and return
    sexAssig = []

    # Hard coded variables
    # most species are bisexual: Female vs Male
    nbSexToPred = 2
    # gonosomes list to cover different species
    gonoChromList = ["X", "Y", "W", "Z"]
    # empty status to assign a boolean if the data comes from mammals => True
    # or others => False
    mammals = None

    #####
    # species identification
    # Know which gonosomes operate if X and Y => mammals,
    # if Z and W => others
    # check that the chromosome pairs are correct if not the assignment cannot be made
    gonoNames = list(set([exon[0] for exon in gonosomesExons]))
    if gonoNames[0].startswith("chr"):
        gonoChromList = ["chr" + letter for letter in gonoChromList]

    if sorted(gonoNames) == sorted(gonoChromList[:2]):
        mammals = True
    elif sorted(gonoNames) == sorted(gonoChromList[2:]):
        mammals = False
    else:
        logger.error("No predefined gonosomes are present in the exon list (X, Y, Z, W ). Please check that the original BED file.")
        raise Exception()

    #####
    # Performs an empirical method (kmeans) to dissociate male and female.
    # The classical EM-style algorithm used by default is "lloyd"
    kmeans = sklearn.cluster.KMeans(n_clusters=nbSexToPred).fit(gonosomesFPM.T).predict(gonosomesFPM.T)

    #####
    # ratios per Kmeans group and per gonosome calculated
    ratioGono = getRatioGono2KmeanGp(gonoNames, kmeans, gonosomesFPM, gonosomesExons)

    #####
    # sexe assignment for each kmeans groups
    if mammals:
        pred = sexAssignmentPrivate(ratioGono, gonoNames, "Y", "X", ["M", "F"])
    else:
        pred = sexAssignmentPrivate(ratioGono, gonoNames, "Z", "W", ["F", "M"])

    #####
    # fill output results
    for indPred in range(len(pred)):
        samps = [samples[i] for i in range(len(kmeans)) if kmeans[i] == indPred]
        listToFill = [pred[indPred], samps]
        sexAssig.append(listToFill)

    return sexAssig


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
# getRatioGono2KmeanGp
# cover ratio = median(sums of samples counts in a Kmeans group) for one gonosome
# Args:
# - gonoNames (list[str])
# - kmeans (list[int]): sampsNB length where for each sample index in 'samples'
# is associated a group predicted by Kmeans (always 0 or 1)
# - gonosomesFPM (np.ndarray(float))
# - gonosomesExons (list of lists[str])
# Returns a np.ndarray[floats] contains the cover ratio, dim= GonosomesNB * kmeanGpNB
def getRatioGono2KmeanGp(gonoNames, kmeans, gonosomesFPM, gonosomesExons):
    # sorting of Kmeans group identifiers from 0
    groupList = sorted(np.unique(kmeans))

    # To Fill and returns
    ratioGono = np.empty((len(gonoNames), len(groupList)), dtype=np.float32)

    for gonoIndex in range(len(gonoNames)):
        gonoName = gonoNames[gonoIndex]
        for kmeanGroup in groupList:
            #####################
            # selection normalized count data on current gonosome exons (row) and samples
            # in current Kmeans group (column)
            exonsIndex = [i for i, exon in enumerate(gonosomesExons) if exon[0] == gonoName]
            sampsIndexKGp = np.where(kmeans == kmeanGroup)[0]
            gonoTmpArray = gonosomesFPM[exonsIndex][:, sampsIndexKGp]

            #####################
            # cover ratio calculation (axis=0, sum all exons for each sample)
            ratioGono[gonoIndex, kmeanGroup] = np.median(np.sum(gonoTmpArray, axis=0))

    return ratioGono


#############################################################
# sexAssignmentPrivate
# Two independent predictions on the coverage ratios are performed:
# 1) identification of sexe with a specific gonosome(eg: human Male= chrY)
# One group is expected to have a 10 x higher ratio of this gonosome than the other sexe.
# 2) identification of sexe with two identical gonosomes (eg: human Female = 2*chrX)
# One groups(G1) is expected to have : 1.5*ratio(G2)<ratio(G1)< 3*ratio(G2) for this gonosome.
# Args:
# - ratioGono (np.ndarray[floats]): cover ratio, dim = gonosomesNB * kmeansGpNB
# - gonoNames (list[str])
# - chrSexUniq [str]: name of the single sex-specific gonosome (eg: human Male= "chrY", bird Female= "chrW")
# - chrSexDup [str]: name of the duplicated sex-specific gonosome (eg: human Female= "chrX", bird Male= "chrZ")
# - sexNames (list[str]): names of the sexes in order: 1) sex with a single gonosome 2) sex with a double gonosome
# (eg: human ["M","F"], bird ["F","M"])
# returns a list[str] of sex "F", "M" in the order of the Kmeans groups [0,1] in case
# the two predictions agree otherwise returns an exception and the assignment cannot be performed
def sexAssignmentPrivate(ratioGono, gonoNames, chrSexUniq, chrSexDup, sexNames):
    # To Fill and return
    pred1 = []
    pred2 = []

    for i in range(len(ratioGono)):
        # first prediction
        # the first group has a ratio greater than 10x the second group
        # => the first group is associated with the sex with a single specific gonosome
        # if not, reverse the list
        if ((gonoNames[i] == "chr" + chrSexUniq) or (gonoNames[i] == chrSexUniq)):
            if (ratioGono[i][0] > (ratioGono[i][1] * 10)):
                pred1 = sexNames[::-1]  # allows to create a new list without touching the original list
            else:
                pred1 = sexNames
        # second prediction
        # the first group has a ratio between 1.5x and 3x the ratio of the second group
        # => the first group is associated with the sex with the two identical gonosomes,
        # if not, reverse the list
        if ((gonoNames[i] == "chr" + chrSexDup) or (gonoNames[i] == chrSexDup)):
            if (ratioGono[i][0] > (3 * ratioGono[i][1] / 2)) and (ratioGono[i][0] < (3 * ratioGono[i][1])):
                pred2 = sexNames[::-1]
            else:
                pred2 = sexNames

    # check that the two predictions are in agreement to secure the assignment.
    # If not, returns an error message and an exception
    if (pred1 == pred2):
        return pred1

    else:
        logger.error("gender predictions are not in agreement.\n \
            condition n°1, one gender is characterised by a specific gonosome: %s \n \
                condition n°2 that the other gender is characterised by 2 same gonosome copies: %s ", pred1, pred2)
        raise Exception()
