import sys
import os
import numpy as np
import logging

# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###############################################################################
# getGenderInfos:
# - identifies the sex chromosomes (gonosomes) operating in the input data.
# Please note that in this script only the X, Y, Z and W chromosomes can be found.
# This is largely sufficient to cover a large part of the living world
# => mammals, birds, fish, reptiles.
# - associates the expected combinations for each gender.
# The number of genus is limited to 2 for male and female.
# - finds the exons indexes associated with the gonosomes
#
# Args:
#  - exons (list of lists[str,int,int,str]): information on exon, containing CHR,START,END,EXON_ID
#
# Returns a tuple (gonoIndexDict, gendersInfos), each are created here:
#   - gonoIndexDict (dict([str]:list[int])): associated the gonosome names with the
#  corresponding exon indexes
#   - gendersInfos (list of lists[str]): information for identified genders,
#   dim=["gender identifier","particular chromosome"]*2
#   Warning: The indexes of the different lists are important:
#   index 0: gender carrying a unique gonosome not present in the other gender (e.g. human => M:XY)
#   index 1: gender carrying two copies of the same gonosome (e.g. human => F:XX)
def getGenderInfos(exons):
    # step 1: identification of target gonosomes and extraction of associated exon indexes.
    gonoChromList = ["X", "Y", "W", "Z"]
    if exons[0][0].startswith("chr"):
        gonoChromList = ["chr" + letter for letter in gonoChromList]

    gonoIndexDict = {}
    for chr in gonoChromList:
        gono_indexes = [i for i, exon in enumerate(exons) if exon[0] == chr]
        if gono_indexes:
            gonoIndexDict[chr] = gono_indexes
    sortKeyGonoList = sorted(gonoIndexDict.keys())

    # step 2: deduction of the gender list to be used for the analyses.
    if sortKeyGonoList == gonoChromList[:2]:
        # Human case:
        # index 0 => Male with unique gonosome chrY
        # index 1 => Female with 2 chrX
        genderInfoList = [["M", sortKeyGonoList[1]], ["F", sortKeyGonoList[0]]]
    elif sortKeyGonoList == gonoChromList[2:]:
        # Reptile case:
        # index 0 => Female with unique gonosome chrW
        # index 1 => Male with 2 chrZ
        genderInfoList = [["F", sortKeyGonoList[0]], ["M", sortKeyGonoList[1]]]
    else:
        logger.error("No predefined gonosomes are present in the exon list (X, Y, Z, W ).\n \
        Please check that the original BED file containing the exon information matches the gonosomes processed here.")
        raise Exception()
    return gonoIndexDict, genderInfoList


###############################################################################
# genderAttribution:
# Gender matching to groups predicted by Kmeans
# calcul of normalized count ratios per gonosomes and kmeans group
# ratio = median (normalized count sums list for a gonosome and for all samples in a kmean group)
#
# Args:
# - kmeans (list[int]): groupID predicted by Kmeans ordered on SOIsIndex
# - gonosomesFPM (np.ndarray[float]): normalized fragment counts for valid samples,
#  exons covered in gonosomes
# - gonoIndexDict (dict([str]:list[int])): associated the gonosome names with the
#  corresponding exon indexes
# - gendersInfos (list of lists[str]): information for identified genders,
#   dim=["gender identifier","particular chromosome"]*2
#
# Returns:
# - Kmeans2Gender (list[str]): genderID (e.g ["M","F"]), the order
# correspond to KMeans groupID (gp1=M, gp2=F)
def genderAttribution(kmeans, gonosomesFPM, gonoIndex, genderInfo):
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
            gonoTmpArray = gonosomesFPM[gonoIndex[gonoID], ][:, SOIsIndexKGp]

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
    return(sexePredSpeGono)

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
