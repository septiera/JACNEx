import logging
import numpy as np
import scipy.stats
import time
import gzip

# prevent PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
###############################################################################
# exonOnChr:
# Iterates over the exons and assigns a chromosome count to each exon based on
# the order in which the chromosomes appear.
# It keeps track of the previous chromosome encountered and increments the count
# when a new chromosome is found.
#
# Arg:
#  - a list of exons, each exon is a list of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID
#
# Returns an uint8 numpy.ndarray of the same size as exons, value is
# chromosome count for the corresponding exon.
def exonOnChr(exons):
    chrCounter = 0
    exon2Chr = np.zeros(len(exons), dtype=np.uint8)
    prevChr = exons[0][0]

    for idx, exon in enumerate(exons):
        currentChr = exon[0]

        if currentChr != prevChr:
            prevChr = currentChr
            chrCounter += 1

        exon2Chr[idx] = chrCounter

    return exon2Chr


#######################################
# inferCNVsUsingHMM
# Implements the Hidden Markov Model (HMM) algorithm to infer copy number variations (CNVs)
# from a given sample's copy number calls.
# Here's an overview of its steps:
# 1- Create a boolean mask to identify non-called exons in the CNcallOnSamp array.
# 2- Associate each exon with its corresponding chromosome using the exonOnChr function.
# 3- Initialize the path array with -1 as a placeholder for inferred CNVs.
# 4- Iterate over each chromosome present in the dataset.
#     -Create a boolean mask to identify the exons called on the current chromosome.
#     -Apply the Viterbi algorithm using the viterbi function to obtain the inferred path
#      (CNVs) for the called exons.
#     -Assign the obtained path to the corresponding exons in the path array.
# 5- Group exons with the same copy number to obtain the CNVs using the aggregateCalls function.
# Return the resulting CNVs stored in the CNVSampList array.
#
# Args:
# - CNcallOneSamp (np.ndarray[floats]): pseudo emission probabilities (likelihood) of each state for each observation
#                                          for one sample.
#                                          dim = [NbStates, NbObservations]
# - exons (list of lists[str, int, int, str]): A list of exon information [CHR,START,END, EXONID]
# - transMatrix (np.ndarray[floats]): transition probabilities between states. dim = [NbStates, NbStates].
# - priors (numpy.ndarray): Prior probabilities.
#
# Returns:
#     numpy.ndarray: CNV informations [CNtype, startExonIndex, endExonIndex]
def inferCNVsUsingHMM(CNcallOneSamp, exons, transMatrix, priors):
    # Create a boolean mask for non-called exons [-1]
    exNotCalled = np.any(CNcallOneSamp == -1, axis=1)

    # Create a numpy array associating exons with chromosomes (counting)
    exon2Chr = exonOnChr(exons)

    # Initialize the path array with -1
    path = np.full(len(exons), -1, dtype=np.int8)

    CNVSampArray = np.empty((0, 3), dtype=np.int)

    # Iterate over each chromosome
    for thisChr in range(exon2Chr[-1] + 1):
        print(thisChr)
        # Create a boolean mask for exons called on this chromosome
        exonsCalledThisChr = np.logical_and(~exNotCalled, exon2Chr == thisChr)
        exonsInfThisChr = [sublist for sublist, m in zip(exons, exonsCalledThisChr) if m]
        if len(exonsInfThisChr) != 0:
            # Get the path for exons called on this chromosome using Viterbi algorithm
            getPathThisChr = viterbi(CNcallOneSamp[exonsCalledThisChr], priors, transMatrix, exonsInfThisChr)
            # Assign the obtained path to the corresponding exons
            path[exonsCalledThisChr] = getPathThisChr
        else:
            continue  # no callable exons (e.g female no CNVs on chrY)

        # group exons with same CN to obtain CNVs
        CNVExIndSamp = aggregateCalls(path)

        # Create a column with chr index
        chrIndexColumn = np.full((CNVExIndSamp.shape[0], 1), thisChr, dtype=CNVExIndSamp.dtype)

        # Concatenate CNVExIndSamp with chrIndexColumn
        CNVExIndSampChr = np.hstack((CNVExIndSamp, chrIndexColumn))

        CNVSampArray = np.vstack((CNVSampArray, CNVExIndSampChr))

    return CNVSampArray


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# viterbi
# Implements the Viterbi algorithm to compute the most likely sequence of hidden states given observed emissions.
# It initializes the dynamic programming matrix, propagates scores through each observation, and performs
# backtracking to find the optimal state sequence. Additionally, it incorporates a weight ponderation based
# on the length of exons.
# This weight is applied to transitions between exons, favoring the normal number of copies (CN2) and reducing
# the impact of transitions between distant exons.
#
# Args:
# - CNcallOneSamp (np.ndarray[floats]): pseudo emission probabilities (log10_likelihood) of each state for each observation
#                                       for one sample
#                                       dim = [NbObservations, NbStates]
# - priors (np.ndarray[floats]): initial probabilities of each state
# - transMatrix (np.ndarray[floats]): transition probabilities between states. dim = [NbStates, NbStates]
# - exonsInfThisChr (list of lists[str, int, int, str]): A list of exon information for one chromosome
#                                                        [CHR,START,END, EXONID]
#
# Returns:
# - bestPath (list[int]): the most likely sequence of hidden states given the observations and the HMM parameters.
def viterbi(CNcallOneSamp, priors, transMatrix, exonsInfThisChr):
    # Fixed parameters
    # constant value used to normalise the distance between exons
    expectedCNVLength = 1.e8
    # Get the dimensions of the input matrix
    NbObservations, NbStates = CNcallOneSamp.shape

    # Transpose the input matrix in the same format as the emission matrix in the classic Viterbi algorithm
    # dim = [NbStates * NbObservations]
    CNcallOneSamp = CNcallOneSamp.transpose()

    # Step 1: Initialize variables
    # The algorithm starts by initializing the dynamic programming matrix "pathProbs" and the "path" matrix.
    # pathProbs: stores the scores of state sequences up to a certain observation,
    # path: is used to store the indices of previous states with the highest scores.
    pathProbs = np.full((NbStates, NbObservations), -np.inf, dtype=np.float128)
    path = np.zeros((NbStates, NbObservations), dtype=np.uint8)

    # Step 2: Fill the first column of path probabilities with prior values
    pathProbs[:, 0] = priors + CNcallOneSamp[:, 0]

    # Keep track of the end position of the previous exon
    previousExEnd = exonsInfThisChr[0][2]

    # Step 3: Score propagation
    # Iterate through each observation, current states, and previous states.
    # Calculate the score of each state by considering:
    #  - the scores of previous states
    #  - transition probabilities (weighted by the distance between exons)
    for obNum in range(1, NbObservations):
        # Get the start position of the current exon and deduce the distance between the previous and current exons
        currentExStart = exonsInfThisChr[obNum][1]
        exDist = currentExStart - previousExEnd

        for state in range(NbStates):
            # Weighting for transitions between exons based on their distance
            # - Greater distances are more penalized
            # - Transition favoring the normal number of copies (CN2) is not weighted
            # - No weighting for overlapping exons
            # This weight is used to adjust the transition probabilities between states.
            weight = 0
            if (state != 2) and (exDist > 0):
                weight = -exDist / expectedCNVLength

            probMax = -np.inf
            prevMax = -1

            for prevState in range(NbStates):
                pondTransition = (transMatrix[prevState, state] + weight)
                # Calculate the probability of observing the current observation given a previous state
                prob = pathProbs[prevState, obNum - 1] + pondTransition + CNcallOneSamp[state, obNum]
                # Find the previous state with the maximum probability
                # Update the value of the current state ("pathProbs")
                # Store the index of the previous state with the maximum probability in the "path" matrix
                if prob > probMax:
                    probMax = prob
                    prevMax = prevState

            # Store the maximum probability and CN type index for the current state and observation
            pathProbs[state, obNum] = probMax
            path[state, obNum] = prevMax

        previousExEnd = exonsInfThisChr[obNum][2]

    # Step 4: Backtracking the path
    # Retrieve the most likely sequence of hidden states by backtracking through the "path" matrix.
    # Start with the state with the highest score in the last row of the "pathProbs" matrix.
    # Follow the indices of previous states stored in the "path" matrix to retrieve the previous state at each observation.
    bestPath = np.full(NbObservations, -1, dtype=np.uint8)
    bestPath[NbObservations - 1] = pathProbs[:, NbObservations - 1].argmax()
    for obNum in range(NbObservations - 1, 0, -1):
        bestPath[obNum - 1] = path[bestPath[obNum], obNum]

    return bestPath


######################################
# aggregateCalls
# Groups the CNVs (Copy Number Variants) based on the specified conditions.
# It then performs analysis on special cases and updates the boundaries of certain CNVs accordingly.
# Finally, it returns an array CNVList containing the grouped and updated CNVs.
#
# Args:
# - path (list[int]): the most likely sequence of hidden states given the observations and the HMM parameters.
#
# Returns:
# - CNVList (np.ndarray[ints]): CNV informations [CNtype, startExonIndex, endExonIndex]
def aggregateCalls(path):
    CNVList = []
    startExId = None
    prevCN = None

    # Group CNVs based on path
    for i in range(len(path)):
        currCN = path[i]
        # Exclude CNs corresponding to no call exons and diploids (CN2, normal)
        if currCN == -1 or currCN == 2:
            # If the previous CNV exists, store the results and reset start
            if startExId is not None:
                CNVList.append([prevCN, startExId, endExId])
                startExId = None
            continue
        # CNV case: CN0, CN1, CN3
        else:
            # First occurrence, store information of the first exon of the CNV
            if startExId is None:
                startExId = i
                endExId = i
                prevCN = currCN
            # Difference between CNs, store results of the previous CNV and update information of the current CNV
            elif currCN != prevCN:
                CNVList.append([prevCN, startExId, endExId])
                startExId = i
                endExId = i
                prevCN = currCN
            # Same CN, update the end
            else:
                endExId = i

    # Consider the last iteration
    if startExId is not None:
        CNVList.append([prevCN, startExId, endExId])

    # Special cases:
    # If a Homozygous Variant (HV) is followed by a Heterozygous Variant (HET) or vice versa,
    # extend the HET CNV to the boundaries of the HV CNV.
    # For Duplication CNVs (DUP), we do not want to change the boundaries if it has surrounding deletions.
    for i in range(1, len(CNVList)):
        if CNVList[i][0] == 1 and CNVList[i - 1][0] == 0:
            CNVList[i][1] = CNVList[i - 1][1]

        if CNVList[i][0] == 0 and CNVList[i - 1][0] == 1:
            CNVList[i - 1][2] = CNVList[i][2]

    return np.array(CNVList)


##############
# CNV2Vcf
# Takes an array of CNV information, an array of exon information, and a
# list of sample names as inputs.
# It sorts the array of CNV informatio based on specific columns using np.lexsort.
# Then, it iterates through the sorted array to create a VCF output list
#
# Args:
# - CNVArray (np.ndarray[ints]) : an array of CNV information [CN,START,END,CHR,SAMPID]
# - exons (list of lists[str, int, int, str]): A list of exon information [CHR,START,END, EXONID]
# - samples (list[strs]): sample names
# - padding [int]: user defined parameters (used in s1_countFrags.py)
#
# Returns:
# - a list of lists (List[List[Any]]) representing the VCF output.
#   Each inner list contains the information for a single VCF line.
def CNV2Vcf(CNVArray, exons, samples, padding):
    NBnonSampleColumns = 9  # ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]
    # Sort the array based on multiple columns in a specific order
    # filtering step: END, START, CN, CHR
    sorted_array = CNVArray[np.lexsort((CNVArray[:, 2], CNVArray[:, 1], CNVArray[:, 0], CNVArray[:, 3]))]

    vcf = []  # output list

    # Dictionary to store CNV information by key (chrom, pos, end, type)
    cnv_dict = {}

    for cnvIndex in range(len(sorted_array)):
        cnvInfo = sorted_array[cnvIndex]
        chrom = exons[cnvInfo[1]][0]  # str
        # remove padding
        pos = exons[cnvInfo[1]][1] + padding  # int
        end = exons[cnvInfo[2]][2] - padding  # int
        CNtype = cnvInfo[0]  # int
        currentCNV = (chrom, pos, end, CNtype)

        # CNV with the same key already exists in the dictionary,
        # update the corresponding sample's value in the VCF line
        if currentCNV in cnv_dict:
            vcfLine = cnv_dict[currentCNV]
            # Get the index of the sample in the VCF line
            # (+9 to account for the non-sample columns)
            sampleIndex = cnvInfo[3] + NBnonSampleColumns
            # Determine the sample's genotype based on CNV type
            sampleInfo = "1/1" if CNtype == 0 else "0/1"
            vcfLine[sampleIndex] = sampleInfo

        # CNV location and type not seen before, create a new VCF line
        else:
            if (CNtype != 3):
                vcfLine = [chrom, pos, ".", ".", "<DEL>", ".", ".", "SVTYPE=DEL;END=" + str(end), "GT"]
            else:
                vcfLine = [chrom, pos, ".", ".", "<DUP>", ".", ".", "SVTYPE=DUP;END=" + str(end), "GT"]
            # Set CN2 default values for all samples columns
            sampleInfo = ["0/0"] * len(samples)
            sampleIndex = cnvInfo[3] + NBnonSampleColumns
            sampleInfo[sampleIndex] = "1/1" if CNtype == 0 else "0/1"
            vcfLine += sampleInfo
            # Store the VCF line in the dictionary for future reference
            cnv_dict[currentCNV] = vcfLine
            vcf.append(vcfLine)

    return vcf


##########################################
# printVcf
#
# Args : 
# - vcf (list of lists)
# - outFile [str]: filename that doesn't exist, it can have a path component (which must exist),
#                  output will be gzipped if outFile ends with '.gz'
# - scriptName [str]
# - samples (list[strs]): sample names.
def printVcf(vcf, outFile, scriptName, samples):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open CNCallsFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open CNCallsFile')
    
    # Header definition
    toPrint = """##fileformat=VCFv4.3
    ##fileDate=""" + time.strftime("%y%m%d") + """
    ##source=""" + scriptName + """
    ##ALT=<ID=DEL,Description="Deletion">
    ##ALT=<ID=DUP,Description="Duplication">
    ##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
    ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype (always 0/1 for duplications)">"""
    toPrint += "\n"
    outFH.write(toPrint)

    colNames = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
    print('\t'.join(colNames))
    
    #### fill results
    for cnvIndex in range(len(vcf)):
        toPrint = '\t'.join(str(x) for x in vcf[cnvIndex])
        toPrint += "\n"
        outFH.write(toPrint)
    outFH.close()
