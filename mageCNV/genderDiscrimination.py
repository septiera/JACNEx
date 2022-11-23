import sys
import os
import numpy as np
import time
import logging

# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))

###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###############################################################################
# getGenderInfos: 
# From a list of exons, identification of gonosomes and genders.
# These gonosomes are predefined and limited to the X,Y,Z,W chromosomes present
# in most species (mammals, birds, fish, reptiles).
# The number of genders is therefore limited to 2, i.e. Male and Female
# Arg:
#  - exons: is a list of exons, ie each exon is a lists of 4 scalars 
#           (types: str,int,int,str) containing CHR,START,END,EXON_ID
# Returns a tuple (gonoIndexDict, gendersInfos), each are created here:
#   - gonoIndexDict: is a dictionary where key=GonosomeID(e.g 'chrX')[str], 
#                   value= gonosome exon index list [int].
#   - gendersInfos: is a str list of lists, contains informations for the gender
#                   identification, ie ["gender identifier","particular chromosome"].
# The indexes of the different lists are important:
# index 0: gender carrying a unique gonosome not present in the other gender (e.g. human => M:XY)
# index 1: gender carrying two copies of the same gonosome (e.g. human => F:XX) 

def getGenderInfos(exons):
    # pre-defined list of gonosomes 
    # combinations: X with Y and Z with W + alphabetical order
    gonoChromList = ["X", "Y", "W", "Z"]
    # reading the first line of "exons" for add "chr" to 'gonoChromList' in case 
    # of this line have it
    if (exons[0][0].startswith("chr")):
        gonoChromList = ["chr" + letter for letter in gonoChromList]
    
    # add exon index in int list value for the 
    # correspondant gonosome identifier (str key). 
    gonoIndexDict = {}
    for exonIndex in range(len(exons)):
        if (exons[exonIndex][0] in gonoChromList):
            if (exons[exonIndex][0] in gonoIndexDict):
                gonoIndexDict[exons[exonIndex][0]].append(exonIndex)
            # key initialization
            else:
                gonoIndexDict[exons[exonIndex][0]] = [exonIndex]
        # exon in an autosome
        # no process next
        else:
            continue
            
    # the dictionary keys may not be sorted alphabetically
    # needed to compare with gonoChromList      
    sortKeyGonoList = list(gonoIndexDict.keys())
    sortKeyGonoList.sort()
    if (sortKeyGonoList==gonoChromList[:2]):
        # Human case:
        # index 0 => Male with unique gonosome chrY
        # index 1 => Female with 2 chrX 
        genderInfoList = [["M",sortKeyGonoList[1]],["F",sortKeyGonoList[0]]]
    elif (sortKeyGonoList==gonoChromList[2:]):
        # Reptile case:
        # index 0 => Female with unique gonosome chrW
        # index 1 => Male with 2 chrZ 
        genderInfoList = [["F",sortKeyGonoList[0]],["M",sortKeyGonoList[1]]]
    else:
        logger.error("No predefined gonosomes are present in the exon list (X, Y, Z, W ).\n \
        Please check that the exon file initially a BED file matches the gonosomes processed here.")
        raise Exception
    return(gonoIndexDict, genderInfoList)

###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

###############################################################################
# genderAttributionPrivate: [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Gender matching to groups predicted by Kmeans 
# calcul of normalized count ratios per gonosomes and kmeans group
# ratio = median (normalized count sums list for a gonosome and for all samples in a kmean group)
# Args:
#  -kmeans: an int list of groupID predicted by Kmeans ordered on SOIsIndex 
#  -countsNorm: a float 2D numpy array of normalized count, dim=NbExons*NbSOIs
#  -gonoIndexDict: a dictionary of correspondence between gonosomeID[key:str] and exonsIndex[value:int list] 
#  -genderInfoList: a list of list, dim=NbGender*2columns ([genderID, specificChr])
# Returns a list of genderID where the indexes match the groupID formed by the Kmeans.
# e.g ["F","M"], KmeansGp 0 = Female and KmeansGp 1 = Male

def genderAttributionPrivate(kmeans, countsNorm, gonoIndex, genderInfo):
    # first browse on gonosome names (limited to 2)
    for gonoID in gonoIndex.keys():
        # previousCount: a float variable, store the count ratio of the first Kmeans group
        # for a given gonosome
        previousCount = None

        # second browse on the kmean groups (only 2)
        for kmeanGroup in np.unique(kmeans.labels_):
            # SOIsIndexKGp: an int list corresponding to the sample indexes in the current Kmean group
            SOIsIndexKGp,=list(np.where(kmeans.labels_ == kmeanGroup))
            
            #####################
            # selection of specifics normalized count data
            gonoTmpArray = countsNorm[gonoIndex[gonoID],] # gonosome exons
            gonoTmpArray = gonoTmpArray[:,SOIsIndexKGp] # Kmean group samples        
            
            #####################
            # ratio calcul (axis=0, sum all row/exons for each sample) 
            countRatio = np.median(np.sum(gonoTmpArray,axis=0))

            # Keep gender names in str variables
            g1 = genderInfo[0][0] #e.g human g1="M"
            g2 = genderInfo[1][0] #e.g human g2="F"
            
            #####################
            # filling two 1D string list (e.g ["M","F"]) 
            # indexes corresponds to Kmeans groupID 
            # sexePredSpeGono: prediction made on the ratio obtained from the gonosome 
            # gender-specific (e.g human chrY, reptile chrW)
            # sexePredDupGono: prediction made on the ratio obtained from the duplicated 
            # gonosome for a gender (e.g human chrX, reptile chrZ)
            if (previousCount!=None):    
                # row order in genderInfo is important
                # 0 : gender with specific gonosome not present in other gender
                # 1 : gender with 2 copies of same gonosome
                if (gonoID == genderInfo[0][1]): # e.g human => M:chrY
                    #####################
                    # condition assignment gender number 1:
                    # e.g human case group of M, the chrY ratio is expected 
                    # to be higher than the group of F (>10*)
                    countsx10 = 10*countRatio
                    sexePredSpeGono = [""]*2
                    if (previousCount>countsx10):
                        sexePredSpeGono[kmeanGroup-1] = g1 #e.g human Kmeans gp0 = M
                        sexePredSpeGono[kmeanGroup] = g2 #e.g human Kmeans gp1 = F
                    else:
                        sexePredSpeGono[kmeanGroup-1] = g2 #e.g human Kmeans gp0 = F
                        sexePredSpeGono[kmeanGroup] = g1 #e.g human Kmeans gp1 = M
                else: # e.g human => F:chrX
                    #####################
                    # condition assignment gender number 2:
                    # e.g human case group of F, the chrX ratio should be in 
                    # the range of 1.5*ratiochrXM to 3*ratiochrXM
                    countsx1half=3*countRatio/2
                    sexePredDupGono=[""]*2
                    if ((previousCount>countsx1half) and (previousCount<2*countsx1half)):
                        sexePredDupGono[kmeanGroup-1] = g2
                        sexePredDupGono[kmeanGroup] = g1
                    else:
                        sexePredDupGono[kmeanGroup-1] = g1
                        sexePredDupGono[kmeanGroup] = g2
            else:
                # It's the first ratio calculated for the current gonosome => saved for comparison 
                # with the next ratio calcul
                previousCount = countRatio
            
    # predictions test for both conditions
    # the two lists do not agree => raise an error and quit the process.
    if (sexePredSpeGono!=sexePredDupGono):
        logger.error("The conditions for gender prediction are not in agreement.\n \
            condition n°1, one gender is characterised by a specific gonosome: %s \n \
                condition n°2 that the other gender is characterised by 2 same gonosome copies: %s ", sexePredSpeGono, sexePredDupGono)
        raise Exception
    return(sexePredSpeGono)

