#!/usr/bin/env python3
# coding: utf-8

#############################################################################################################
############# STEP3 VCF Formatting
#############################################################################################################
# How the script works ?
#This script allows to format the results of CNV calls obtained with Bedtools/ExomeDepth in VCF.
#Several steps are required for this formatting:
#-checking the format of the input file.
#Pre-processing of the data (removal of padding, filtering on the BF, obtaining the copy number column)
#-Definition of a hash table allowing an optimal data processing (key=chromosome:start_end; value=sample_CN_BayesFactor_ReadsRatio)
#-Formatting
#-Addition of the vcf header and saving.

#The input parameters:
#-the path to the .tsv file of the ExomeDepth output.
#-the path to save the output vcf.
#-Bayes factor filtering threshold (all CNVs with a lower BF level will be removed from the analysis).

#The output file must respect the vcf v4.3 format.
#It can then be annotated by the VEP software.

#############################################################################################################
############# Loading of the modules required for processing.
#############################################################################################################

import pandas as pd #is a module that makes it easier to process data in tabular form by formatting them in dataframes as in R.
import numpy as np #is a module often associated with the pandas library it allows the processing of matrix or tabular data.
import os #this module provides a portable way to use operating system-dependent functionality. (opening and saving files)
os.environ['NUMEXPR_NUM_THREADS'] = '10' #allows to define a CPUs minimum number to use for the process. Must be greater than the cores number defined below.
import sys, getopt #this module provides system information. (ex argv argument)
import time #is a module for obtaining the date and time of the system.
import logging #is a practical logging module. (process monitoring and development support)
import fnmatch, re #this module allows you to search and replace characters with regular expressions. (similar to "re" module)
import multiprocessing #this module allows to parallelize the process.
from joblib import Parallel, delayed #this module allowing to define the parallelization characteristics works with the multiprocessing module.

#Unique Modules Localization necessary to avoid version errors between different processes.
sys.path.append("/home/septiera/Benchmark_CNVTools/Scripts/Modules/")
import FileFolderManagement.functions as ffm #Module to check the existence of files
import InputSanityCheck.functions as isc #Sanity Check of input files

#Scripts execution date definition(allows to annotate the output files to track them over time) 
now=time.strftime("%y%m%d")

#CPU number definition to use during parallelization.
num_cores=5

pd.options.mode.chained_assignment = None
#############################################################################################################
############# Logging Definition 
#############################################################################################################
#create logger
logger = logging.getLogger(sys.argv[0])
logger.setLevel(logging.DEBUG)
#create console handler and set level to debug
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
#create formatter
formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)-8s [%(process)d] %(message)s', '%Y-%m-%d %H:%M:%S')
#add formatter to ch
ch.setFormatter(formatter)
#add ch to logger
logger.addHandler(ch)
##############################################################################################################
############### Script Body
##############################################################################################################

def main(argv):
    ##########################################
    # A) ARGV parameters definition
    callResultsPath = ''
    bayesThreshold=''
    outputFile =''

    try:
        opts, args = getopt.getopt(argv,"h:c:b:o:",["help","callsFile=","BFthreshold=","outputfile="])
    except getopt.GetoptError:
        print('python3.6 STEP_3_VCFFormatting_Bedtools.py -c <callsFile> -b <BFthreshold> -o <outputfile>')
        sys.exit(2)
    for opt, value in opts:
        if opt == '-h':
            print("COMMAND SUMMARY:"			
            +"\n This script allows to format the results of CNV calls obtained with Bedtools/ExomeDepth in VCF."
            +"\n"
            +"\n USAGE:"
            +"\n python3.6 STEP_3_VCFFormatting_Bedtools.py -c <callsFile> -b <BFthreshold> -o <outputfile>"
            +"\n"
            +"\n OPTIONS:"
            +"\n	-c : A tsv file obtained in STEP2 (CNV calling results). Please indicate the full path.(15 columns)"
            +"\n	-b : Bayes factor filtering threshold (all CNVs with a lower BF level will be removed from the analysis)."
            +"\n	-o : The path where create the new output file for the current analysis.")
            sys.exit()
        elif opt in ("-c", "--callsFile"):
            callResultsPath=value
        elif opt in ("-b", "--BFthreshold"):
            bayesThreshold=value
        elif opt in ("-o", "--outputfile"):
            outputFile=value

    #Check that all the arguments are present.
    logger.info('CNVs calling results file path is %s', callResultsPath)
    logger.info('BF threshold is %s ', bayesThreshold)
    logger.info('Output file path is %s ', outputFile)

    #####################################################
    # A) Input file format check
    logger.info("Input file format check")
    tabToTreat=isc.CallBedtoolsSanityCheck(callResultsPath,outputFile)
    
    #####################################################
    # B)Filtering according to the Bayes Factor threshold chosen by the user in the function parameters.
    logger.info("Filtering and Dictionnary creation")
    minBF=int(bayesThreshold)
    CNVTableBFfilter=tabToTreat[tabToTreat.BF>minBF]
    logger.info("1) CNVs Number obtained after BF=%s, filtering=%s",minBF,len(CNVTableBFfilter))

    #####################################################
    # C)Appropriate copy numbers allocation according to the read ratios.
    # Warning filtering of deletions with RR>0.75.
    CNVTableBFfilter.reset_index(drop=True, inplace=True)
    CNVTableBFfilter.loc[:,"CN"]=np.repeat(4,len(CNVTableBFfilter))#CN obsolete allow to delete DEL with RR>0.75
    CNVTableBFfilter.loc[CNVTableBFfilter["type"]=="duplication","CN"]=3
    CNVTableBFfilter.loc[CNVTableBFfilter["reads.ratio"]<0.75,"CN"]=1
    CNVTableBFfilter.loc[CNVTableBFfilter["reads.ratio"]<0.25,"CN"]=0
    CNVTableAllFilter=CNVTableBFfilter[CNVTableBFfilter["CN"]!=4]
    logger.info("2) CNVs Number obtained after attribution copy number = %s",len(CNVTableAllFilter))

    #####################################################
    # D) Standardization STEP
    #Creation of a chromosome column not containing chr at the beginning of the term.
    CNVTableAllFilter["#CHROM"]=CNVTableAllFilter["chromosome"].str.replace("chr","")

    #Position correction => padding removal
    CNVTableAllFilter["POS"]=CNVTableAllFilter["start"]+9 #ExomeDepth automatically adds 1 more base to the start.
    CNVTableAllFilter["end2"]=CNVTableAllFilter["end"]-10

    #CNV identifier creation
    CNVTableAllFilter["ID"]=CNVTableAllFilter["chromosome"]+":"+CNVTableAllFilter["POS"].astype(str)+"_"+CNVTableAllFilter["end2"].astype(str)

    #Sample name formatting
    CNVTableAllFilter["sample"]=CNVTableAllFilter["sample"].str.replace(".tsv","")
    
    #####################################################
    # E) Dictionnary Creation and definition of a function to evaluate the presence of NaN in the columns.
    Dict={}
    for index, row in CNVTableAllFilter.iterrows():
        Key=row["ID"]
        Value=row["sample"]+"_"+str(row["CN"])+"_"+str(row["BF"])+"_"+str(row["reads.ratio"])
        Dict.setdefault(Key,[]).append(Value)
    logger.info("3) Number of dictionnary Keys = %s",len(Dict))
    inputs=np.arange(0,len(Dict))

    def isNaN(string):
        return string != string

    #####################################################
    # F) definition of the function allowing the formatting of calling CNVs results.
    #This function takes as input a indices list (here the number of keys in the dictionary).

    def parallelFormatVCF(IndexKey,VCFTab):
        print(IndexKey)
        #extract Key Value
        key=list(Dict.keys())[IndexKey]
        value=Dict[key]
        
        #Init Tmp Tab
        colNames=["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]+list(np.unique(CNVTableAllFilter["sample"]))
        TAB=pd.DataFrame(columns=colNames)
        #######################################
        ## Key processing
        #chromosome treatment
        chrom=re.sub("^chr([\w]{1,2}):[\d]{3,9}_[\d]{3,9}","\\1", key)
        #pos
        pos=re.sub("^chr[\w]{1,2}:([\d]{3,9})_[\d]{3,9}","\\1", key)
        #end
        end=re.sub("^chr[\w]{1,2}:[\d]{3,9}_([\d]{4,9})","\\1", key)
        
        #initialization of the lines to generate.
        line=0
        linedup=None
        #######################################
        ## Value processing
        for i in value:
            info_list=i.split("_")
            #the value under analysis comes from a deleted CNV.
            if int(info_list[1])<2:
                TAB.loc[line,"ID"]=key
                TAB.loc[line,"REF"]="."
                TAB.loc[line,"QUAL"]="."
                TAB.loc[line,"FILTER"]="."
                TAB.loc[line,"#CHROM"]=chrom
                TAB.loc[line,"POS"]=pos
                TAB.loc[line,"ALT"]="<DEL>"
                TAB.loc[line,"INFO"]="SVTYPE=DEL;END="+end
                TAB.loc[line,"FORMAT"]="GT:BF:RR"
                TAB.loc[line,list(np.unique(CNVTableAllFilter["sample"]))]="0/0:0:0"
                # CNV type distinction : Homo-deletion
                if int(info_list[1])==0:
                    TAB.loc[line,info_list[0]]="1/1:"+str(round(float(info_list[2]),2))+":"+str(round(float(info_list[3]),2))
                # CNV type distinction : Hetero-deletion
                else :
                    TAB.loc[line,info_list[0]]="0/1:"+str(round(float(info_list[2]),2))+":"+str(round(float(info_list[3]),2))
            #the value under analysis comes from a duplicated CNV.
            elif int(info_list[1])>2:
                if linedup==None:
                    linedup=line+1
                    TAB.loc[linedup,"ID"]=key
                    TAB.loc[linedup,"REF"]="."
                    TAB.loc[linedup,"QUAL"]="."
                    TAB.loc[linedup,"FILTER"]="."
                    TAB.loc[linedup,"#CHROM"]=chrom
                    TAB.loc[linedup,"POS"]=pos
                    TAB.loc[linedup,"ALT"]="<DUP>"
                    TAB.loc[linedup,"INFO"]="SVTYPE=DUP;END="+end
                    TAB.loc[linedup,"FORMAT"]="GT:CN:BF:RR"
                    TAB.loc[linedup,list(np.unique(CNVTableAllFilter["sample"]))]="0/0:2:0:0"
                    TAB.loc[linedup,info_list[0]]="0/1:"+info_list[1]+":"+str(round(float(info_list[2]),2))+":"+str(round(float(info_list[3]),2))
                else:
                    TAB.loc[linedup,info_list[0]]="0/1:"+info_list[1]+":"+str(round(float(info_list[2]),2))+":"+str(round(float(info_list[3]),2))    
            else:
                logger.error("The CNVs called has a wrong copy number status. Check it. Key %s; Value %s",key,value)
                sys.exit()
        # Temporary table contents evaluation. If it contains for the same interval(key) either DELs or DUPs or both.
        if (TAB.index==line).all() and (isNaN(TAB.loc[line,"#CHROM"])==True):
            TAB=TAB.drop(labels=line,axis=0)
            VCFTab = VCFTab.append(TAB)
        else:
            VCFTab = VCFTab.append(TAB)
        
        return(VCFTab)

    #####################################################
    # G) Executing the function, sorting and saving the vcf.
    logger.info("VCF Formatting")
    results=pd.DataFrame()
    results = Parallel(n_jobs=num_cores)(delayed(parallelFormatVCF)(i,results) for i in inputs)
    results=pd.concat(results, ignore_index = True)

    #sorting
    results["CHR"]=results["#CHROM"]
    results["CHR"]=results["CHR"].str.replace('X', '23')
    results["CHR"]=results["CHR"].str.replace('Y', '24')
    results["CHR"]=results["CHR"].str.replace('M', '25')
    results["CHR"]=results["CHR"].astype(int)
    results["POS"]=results["POS"].astype(int)
    results=results.sort_values(by=["CHR","POS"])
    results=results.drop(columns=["CHR"])    
    
    #Header definition
    header = """##fileformat=VCFv4.3
    ##fileDate="""+now+"""
    ##source="""+sys.argv[0]+"""
    ##reference=file:///seq/references/
    ##ALT=<ID=DEL,Description="Deletion">
    ##ALT=<ID=DUP,Description="Duplication">
    ##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
    ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
    ##FORMAT=<ID=GT,Number=1,Type=Integer,Description="Genotype">
    ##FORMAT=<ID=CN,Number=1,Type=Integer,Description="Copy number genotype for imprecise events">
    ##FORMAT=<ID=BF,Number=1,Type=Float,Description="Bayes Factor from stuctural variant prediction">
    ##FORMAT=<ID=RR,Number=1,Type=Float,Description="Reads ratio of structural variant">
    """

    #saving
    output_VCF =os.path.join(outputFile,"CNVResults_BedtoolsExomeDepth_"+str(len(np.unique(CNVTableAllFilter["sample"])))+"samples_"+now+".vcf")
    with open(output_VCF, 'w') as vcf:
        vcf.write(header)

    results.to_csv(output_VCF, sep="\t", mode='a', index=False)

if __name__ =='__main__':
    main(sys.argv[1:])
    

