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
# genderAttribution:
# Performs a kmeans on the coverages of the exons present in the gonosomes,
# to identify two groups normally associated with the gender.
# Assigns gender to each group of Kmeans based on their coverage ratios.
# ratio = median (normalized count sums list for a gonosome and for all samples in a kmean group)
# Beware two independent tests on the ratios are performed:
# 1) identification of gender with a specific chromosome (eg: human Male= chrY)
# One group is expected to have a 10 x higher ratio of this chromosome than the other gender.
# 2) identification of gender with two identical chromosomes (eg: human Female = 2*chrX)
# One groups is expected to have a 1.5 x ratio greater than the other for this chromosome.
# Each test returns a list of predicted example genders in the order of the Kmeans
# groups (e.g. [0="M", 1="F"]
# If at the end of the process these two tests do not produce the same result,
# then it's impossible to predict gender (return the error and stop the analysis).
#
# Args:
# - validCounts (np.ndarray[float]): normalized fragment counts for valid samples
# - gonoIndex (dict([str]:list[int])): associated the gonosome names with the
#  corresponding exon indexes
# - gendersInfos (list of lists[str]): information for identified genders,
#   dim=["gender identifier","particular chromosome"]*2
#
# Returns:
# - kmeans (list[int]): groupID predicted by Kmeans ordered on SOIsIndex
# - Kmeans2Gender (list[str]): genderID (e.g ["M","F"]), the order
# correspond to KMeans groupID (gp1=M, gp2=F)
def genderAttribution(validCounts, gonoIndex, genderInfo):
    # cutting normalized count data according to gonosomal exons
    # - gonoIndexFlat (np.ndarray[int]): flat gonosome exon indexes list
    gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist])
    # - gonosomesFPM (np.ndarray[float]): normalized fragment counts for valid samples, exons covered
    # in gonosomes
    gonosomesFPM = np.take(validCounts, gonoIndexFlat, axis=0)

    # Performs an empirical method (kmeans) to dissociate male and female.
    # consider only the coverage for the exons present in the gonosomes
    # Kmeans with k=2 (always)
    # - kmeans (list[int]): groupID predicted by Kmeans ordered on SOIsIndex
    kmeans = sklearn.cluster.KMeans(n_clusters=len(genderInfo), random_state=0).fit(gonosomesFPM.T).predict(gonosomesFPM.T)

    # To fill
    # str list, index 0 => kmeansGroup1, index 1 => kmeansGroup2
    # contain the gender predicted by the ratio count ("M" or "F")
    sexePredSpeGono = [""] * 2
    sexePredDupGono = [""] * 2

    # first browse on gonosome names (limited to 2)
    for gonoID in gonoIndex.keys():
        # previousCount: a float variable, store the count ratio of the first Kmeans group
        # for a given gonosome
        previousCount = None
        # second browse on the kmean groups (only 2)
        for kmeanGroup in np.unique(kmeans):
            # SOIsIndexKGp: an int list corresponding to the sample indexes in the current Kmean group
            SOIsIndexKGp, = list(np.where(kmeans == kmeanGroup))

            #####################
            # selection of specifics normalized count data
            gonoTmpArray = validCounts[gonoIndex[gonoID], ]  # gonosome exons
            gonoTmpArray = gonoTmpArray[:, SOIsIndexKGp]  # Kmean group samples

            #####################
            # ratio calcul (axis=0, sum all row/exons for each sample)
            countRatio = np.median(np.sum(gonoTmpArray, axis=0))

            #####################
            # filling two 1D string list (e.g ["M","F"])
            # indexes corresponds to Kmeans groupID
            # sexePredSpeGono: prediction made on the ratio obtained from the gonosome
            # gender-specific (e.g human chrY, reptile chrW)
            # sexePredDupGono: prediction made on the ratio obtained from the duplicated
            # gonosome for a gender (e.g human chrX, reptile chrZ)
            if (previousCount is not None):
                # row order in genderInfo is important
                # 0 : gender with specific gonosome not present in other gender
                # 1 : gender with 2 copies of same gonosome
                # Keep gender names in str variables
                g1, g2 = genderInfo[0][0], genderInfo[1][0]

                if (gonoID == genderInfo[0][1]):  # e.g human => M:chrY
                    #####################
                    # condition assignment gender number 1:
                    # e.g human case group of M, the chrY ratio is expected
                    # to be higher than the group of F (>10*)
                    countsx10 = 10 * countRatio
                    sexePredSpeGono = [""] * 2
                    if (previousCount > countsx10):
                        sexePredSpeGono[kmeanGroup - 1] = g1  # e.g human Kmeans gp0 = M
                        sexePredSpeGono[kmeanGroup] = g2  # e.g human Kmeans gp1 = F
                    else:
                        sexePredSpeGono[kmeanGroup - 1] = g2  # e.g human Kmeans gp0 = F
                        sexePredSpeGono[kmeanGroup] = g1  # e.g human Kmeans gp1 = M
                else:  # e.g human => F:chrX
                    #####################
                    # condition assignment gender number 2:
                    # e.g human case group of F, the chrX ratio should be in
                    # the range of 1.5*ratiochrXM to 3*ratiochrXM
                    countsx1half = 3 * countRatio / 2
                    sexePredDupGono = [""] * 2
                    if ((previousCount > countsx1half) and (previousCount < 2 * countsx1half)):
                        sexePredDupGono[kmeanGroup - 1] = g2
                        sexePredDupGono[kmeanGroup] = g1
                    else:
                        sexePredDupGono[kmeanGroup - 1] = g1
                        sexePredDupGono[kmeanGroup] = g2
            else:
                # It's the first ratio calculated for the current gonosome => saved for comparison
                # with the next ratio calcul
                previousCount = countRatio

    # predictions test for both conditions
    # the two lists do not agree => raise an error and quit the process.
    if (sexePredSpeGono != sexePredDupGono):
        logger.error("The conditions for gender prediction are not in agreement.\n \
            condition n°1, one gender is characterised by a specific gonosome: %s \n \
                condition n°2 that the other gender is characterised by 2 same gonosome copies: %s ", sexePredSpeGono, sexePredDupGono)
        raise Exception()

    return (kmeans, sexePredSpeGono)

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################