import logging
import numpy as np
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


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
######################################
# viterbi
# Implements the Viterbi algorithm to compute the most likely sequence of hidden
# states given observed emissions.
# It initializes the dynamic programming matrix, propagates scores through each
# observation, and performs backtracking to find the optimal state sequence.
#
# Args:
# - CNcallOneSamp (np.ndarray[floats]): pseudo emission probabilities (likelihood)
#                                       of each state for each observation for one sample
#                                       dim = [NbObservations, NbStates]
# - transMatrix (np.ndarray[floats]): transition probabilities between states + void status
#                                     dim = [NbStates +1, NbStates + 1]
#
# Returns:
# - CNVs (list of lists[int,int,int,str]): [CNtype, FirstExonStart, LastExonEnd, CHR] to complete
def viterbi(CNCallOneSamp, transMatrix):
    CNVs = []  # list of lists to return

    # np.ndarray of called exon indexes
    exIndexCalled = np.where(np.all(CNCallOneSamp != -1, axis=1))[0]

    # Get the dimensions of the input matrix
    NbObservations = len(exIndexCalled)
    NbStatesWithVoid = len(transMatrix)

    # control that the right number of hidden state
    if NbStatesWithVoid != CNCallOneSamp.shape[1] + 1:
        logger.error("NbStates not consistent with number of columns + 1 in CNcallOneSamp")
        raise

    # Transpose the input matrix in the same format as the emission matrix in the classic
    # Viterbi algorithm, dim = [NbStates * NbObservations]
    CNCallOneSamp = CNCallOneSamp.transpose()

    # Step 1: Initialize variables
    # probsPrev[i]: stores the probability of the most likely path ending in state i at the previous exon
    probsPrev = np.zeros(NbStatesWithVoid, dtype=np.float128)
    probsPrev[0] = 1
    # probsCurrent[i]: same ending at current exon
    probsCurrent = np.zeros(NbStatesWithVoid, dtype=np.float128)
    # path[i,e]: state at exon e-1 that produces the highest probability ending in state i at exon e
    # (ie, probsCurrent[i])
    # this state is 0 (void) if the path starts here
    path = np.zeros((NbStatesWithVoid, NbObservations), dtype=np.uint8)

    # Step 2: Score propagation
    # Iterate through each observation, current states, and previous states.
    # Calculate the score of each state by considering:
    #  - the scores of previous states
    #  - transition probabilities
    exonIndex = 0
    while exonIndex < len(exIndexCalled):

        for currentState in range(NbStatesWithVoid):
            probMax = -1
            prevStateMax = -1

            # state 0 == void is never in a path except at initialisation/reset => keep defaults
            if currentState == 0:
                probsCurrent[currentState] = 0
                path[currentState, exonIndex] = 0
                continue

            else:
                for prevState in range(NbStatesWithVoid):
                    # Calculate the probability of observing the current observation given a previous state
                    prob = (probsPrev[prevState] *
                            transMatrix[prevState, currentState] *
                            CNCallOneSamp[currentState - 1, exIndexCalled[exonIndex]])
                    # currentState - 1 because CNCallOneSamp doesn't have values for void state

                    # Find the previous state with the maximum probability
                    # Store the index of the previous state with the maximum probability in the "path" matrix
                    if prob >= probMax:
                        probMax = prob
                        prevStateMax = prevState

                # Store the maximum probability and CN type index for the current state and observation
                probsCurrent[currentState] = probMax
                path[currentState, exonIndex] = prevStateMax

        # if all LIVE states (currentScore > 0) at currentExon have the same
        # predecessor state and that state is CN2 : backtrack from [previous exon, CN2]
        # and reset
        if (((probsCurrent[1] == 0) or (path[1, exonIndex] == 3)) and
            ((probsCurrent[2] == 0) or (path[2, exonIndex] == 3)) and
            ((probsCurrent[3] == 0) or (path[3, exonIndex] == 3)) and
            ((probsCurrent[4] == 0) or (path[4, exonIndex] == 3))):
            # print("viterbi reset:", exonIndex)
            CNVs.extend(backtrack_aggregateCalls(path, exonIndex - 1, 3))
            # reset prevScores to (1,0,0,0,0), and loop again (WITHOUT incrementing exonIndex)
            probsPrev[1:] = 0
            probsPrev[0] = 1
            path[0, exonIndex - 1] = 0
            path[1:, exonIndex - 1] = 5
            continue

        else:
            for i in range(len(probsCurrent)):
                probsPrev[i] = probsCurrent[i]
            exonIndex += 1

    CNVs.extend(backtrack_aggregateCalls(path, exonIndex - 1, probsPrev.argmax()))

    mapCNVIndicesToExons(CNVs, exIndexCalled)

    return CNVs


################################
# backtrack_aggregateCalls
# Retrieve the most likely sequence of hidden states by backtracking through the "path" matrix.
# aggregation of exons if same CN predicted in CNVs.
#
# Args:
# - path (np.ndarray[unint]): dynamic matrix generated by the Viterbi algorithm, used to store
#                             the optimal state transitions for each time step.
#                             dim = [NBState, NBObservations]
# - lastExon [int]: last exon traversed by the viterbi algorithm before the reset or the end
#                   of the chromosome path.
# - lastState [int]: state with the best probability for the last exon, best path.
#
# Returns a list of CNVs, a CNV == [CNType, startExon, endExon] where CNType is the CN,
# startExon and endExon are the indexes (in "path")
def backtrack_aggregateCalls(path, lastExon, lastState):
    CNVs = []
    startExon = None
    endExon = None
    # retention of an exon index to correctly delimit hetDELs in special cases
    # (see following description in the code)
    hetDelEndExon = None

    # Iterate over exons in reverse order
    nextExon = lastExon
    nextState = lastState

    if nextState != 3:
        endExon = nextExon

    while True:
        currentState = path[nextState, nextExon]

        # currentState == 0, we are at the beginning of the best path,
        # create first CNV(s) if needed
        if currentState == 0:
            if nextState != 3:
                startExon = nextExon
                CNVs.append([nextState, startExon, endExon])
                if hetDelEndExon is not None:
                    CNVs.append([2, startExon, hetDelEndExon])
                    hetDelEndExon = None
            break

        elif currentState == nextState:
            pass

        elif nextState == 3:
            endExon = nextExon - 1

        elif currentState == 3:
            startExon = nextExon
            CNVs.append([nextState, startExon, endExon])
            if hetDelEndExon is not None:
                CNVs.append([2, startExon, hetDelEndExon])
                hetDelEndExon = None

        # DUPs
        elif (currentState == 4) or (nextState == 4):
            startExon = nextExon
            CNVs.append([nextState, startExon, endExon])
            if hetDelEndExon is not None:
                CNVs.append([2, startExon, hetDelEndExon])
                hetDelEndExon = None
            endExon = nextExon - 1

        # Special cases:
        # If a HVDEL is followed by a HETDEL or vice versa,
        # extend the HETDEL to the boundaries of the HVDEL.
        # 1) HETDEL followed by HVDEL
        elif (currentState == 2) and (nextState == 1):
            # create HVDEL
            startExon = nextExon
            CNVs.append([nextState, startExon, endExon])
            # keep endExon except if HVDEL was preceded by another HETDEL,
            # in which case ...
            if hetDelEndExon is not None:
                endExon = hetDelEndExon
                hetDelEndExon = None

        # 2) HVDEL followed by HETDEL
        elif (currentState == 1) and (nextState == 2):
            hetDelEndExon = endExon
            endExon = nextExon - 1
        else:
            raise Exception("not captured case")

        nextState = currentState
        nextExon -= 1

    CNVs.reverse()

    return(CNVs)


############
# mapCNVIndicesToExons
# Map CNV indexes to "exons" indexes based on the 'exIndexCalled' list.
#
# Args:
# CNVs (list of lists): A list containing CNV events, where each event is represented
#                       as a list [start, end, state].
# exIndexCalled (list[int]): A list containing the indexes of exons used in the Viterbi
#                            algorithm's output.
#
# Returns:
# None (The CNVs list will be modified in-place.)
def mapCNVIndicesToExons(CNVs, exIndexCalled):
    for event in range(len(CNVs)):
        CNVs[event][0] = CNVs[event][0] - 1
        CNVs[event][1] = exIndexCalled[CNVs[event][1]]
        CNVs[event][2] = exIndexCalled[CNVs[event][2]]


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
