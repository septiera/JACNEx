#############################################################################################################
###################################### STEP3 VCF Formatting #################################################
#############################################################################################################
# How the script works ?
#This script allows to format the results of CNV calls obtained with ExomeDepth in VCF.
#Several steps are required for this formatting:
#-checking the format of the input file.
#-Pre-processing of the data (padding removal, filtering on the BF, obtaining the copy number column)
#-Definition of a hash table allowing an optimal data processing (key=chromosome:start_end; value=sample_CN_BayesFactor_ReadsRatio)
#-Formatting
#-Addition of the vcf header and saving.

#The input parameters:
#-the path to the .tsv file of the ExomeDepth output.
#-the path to save the output vcf.
#-Bayes factor filtering threshold (all CNVs with a lower BF level will be removed from the analysis).

#The output file must respect the vcf v4.3 format. (cf https://samtools.github.io/hts-specs/VCFv4.3.pdf)
#It can then be annotated by the VEP software.

#############################################################################################################
################################ Loading of the modules required for processing #############################
#############################################################################################################

import pandas as pd # dataframe objects
import numpy as np #is a module often associated with the pandas library it allows the processing of matrix or tabular data.
import os
import sys
import getopt
import time 
import logging #is a practical logging module. (process monitoring and development support)
import re #this module allows you to search and replace characters with regular expressions. 

#Scripts execution date definition(allows to annotate the output files to track them over time) 
now=time.strftime("%y%m%d")

pd.options.mode.chained_assignment = None

#####################################################################################################
################################ Logging Definition #################################################
#####################################################################################################
# set up logger
logger=logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.DEBUG)
# create console handler for STDERR
stderr = logging.StreamHandler(sys.stderr)
stderr.setLevel(logging.DEBUG)
#create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                                '%Y-%m-%d %H:%M:%S')
#add formatter to stderr handler
stderr.setFormatter(formatter)
#add stderr handler to logger
logger.addHandler(stderr)

#####################################################################################################
################################ Functions ##########################################################
#####################################################################################################
####################################################
#CNVs call table sanity check.
#General control : exact number of columns = 15 and colnames identical to a classical 
#ExomeDepth output.
#Columns type controls  
#Input: the path to the ExomeDepth CNVs call table.
#Output: this function returns a dataframe 

def CNVCallSanityCheck(PathToResults):
    filename=os.path.basename(PathToResults)
    if not os.path.isfile(PathToResults):
        logger.error("Result calling CNV file %s doesn't exist.",filename)
        sys.exit(1)
    try:
        cnvDF=pd.read_table(PathToResults,sep="\t")
    except Exception as e:
        logger.error("error parsing Result calling CNV file %s: %s", filename, e)
        sys.exit(1)

    #####################
    #I): General control
    #####################
    if len(cnvDF.columns) != 15:
        logger.error("%s doesn't contains 15 columns as expected.",filename)
        sys.exit(1)

    # check that the column names are the expected ones.
    # Warning to the change of version of ExomeDepth the names are likely to be modified.
    listcolumns=['sample', 'correlation', 'N.comp', 'start.p', 'end.p', 'type',
                 'nexons','start', 'end', 'chromosome', 'id', 'BF', 'reads.expected',
                 'reads.observed', 'reads.ratio']

    if (set(list(cnvDF.columns))!= set(listcolumns)):
        logger.error("%s doesn't contains the correct column names.", filename)
        sys.exit(1)
        
    #######################
    #II) : Columns type controls 
    #######################
    #only the columns useful for processing are checked 
    #######################
    #sample column
    if (cnvDF["sample"].dtype!="O"):
        logger.error("In result calling CNV file %s, 'sample' column should be a string but pandas sees it as %s\n",
                      filename,cnvDF['sample'].dtype)   
        sys.exit(1)

    #######################
    #type column
    if (cnvDF["type"].dtype!="O"):
        logger.error("In result calling CNV file %s, 'type' column should be a string but pandas sees it as %s\n",
                      filename,cnvDF['type'].dtype)   
        sys.exit(1)

    if (set(list(cnvDF["type"].unique()))!=set(["duplication","deletion"])):   
        logger.error("In result calling CNV file %s, 'type' column does not only contain the annotations 'duplication','deletion': %s\n",
                      filename,set(list(cnvDF["type"].unique())))  
        sys.exit(1)

    #######################
    #id column
    if (cnvDF["id"].dtype!="O"):
        logger.error("In result calling CNV file %s, 'id' column should be a string but pandas sees it as %s\n",
                      filename,cnvDF['id'].dtype)   
        sys.exit(1)

    #######################
    #Bayes factor column 
    if (cnvDF["BF"].dtype!=float):
        logger.error("In result calling CNV file %s, columns 'BF' should be floats but pandas sees them as %s\n",
                     filename, cnvDF["BF"].dtype)
        sys.exit(1)

    #######################
    #reads.ratio columns
    if (cnvDF["reads.ratio"].dtype!=float):
        logger.error("In result calling CNV file %s, columns 'BF' should be floats but pandas sees them as %s\n",
                     filename, cnvDF["reads.ratio"].dtype)
        sys.exit(1)

    return(cnvDF)


####################################################
#This function creates one result line in VCF format: 
#"#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SampleCNVStatusFormat"
#Inputs parameters:
#sample= sample name string
#chrom=number or letter corresponding to the chromosome in string format (not preceded by chr)
#pos=cnv start (int)
#key=corresponds to the absolute identifier of the CNV: "CHR:start-end" (string)
#end=cnv end (int)
#CN=copy number (int), BF= Bayes Factor (float), RR= reads ratio (float)
#listS= samples names list
 
def VCFLine(sample,chrom,pos,key,end,CN,BF,RR,listS):
    listResults=["0/0"]*len(listS)      
    #the value under analysis comes from a deleted CNV.
    if int(CN)<2: 
        line=[chrom,pos,key,".","<DEL>",".",".","SVTYPE=DEL;END="+end,"GT:BF:RR"]
        if int(CN)==0: # Homo-deletion
            ValueToAdd=("1/1:"+str(round(float(BF),2))+":"+str(round(float(RR),2)))
            indexToKeep=listS.index(sample)
            listResults[indexToKeep]=ValueToAdd
        elif int(CN)==1: #Hetero-deletion
            ValueToAdd=("0/1:"+str(round(float(BF),2))+":"+str(round(float(RR),2)))
            indexToKeep=listS.index(sample)
            listResults[indexToKeep]=ValueToAdd
    #the value under analysis comes from a duplicated CNV.
    elif int(CN)>2:
        line=[chrom,pos,key,".","<DUP>",".",".","SVTYPE=DUP;END="+end,"GT:CN:BF:RR"]
        ValueToAdd=("0/1:"+str(int(CN))+":"+str(round(float(BF),2))+":"+str(round(float(RR),2)))
        indexToKeep=listS.index(sample)
        listResults[indexToKeep]=ValueToAdd  

    line=line+listResults
    return(line)

#This function allows to complete the lines already generated by the previous function.
#Format: "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "PatientsCNVStatusFormat"
#Inputs parameters:
#sample= sample name string
#CN=copy number (int), BF= Bayes Factor (float), RR= reads ratio (float)
#listS= samples names list
#line= the current line to be completed
 
def VCFLineCompleting(sample,CN,BF,RR,listS,line):
    if (line[4]=="<DEL>"):
            if int(CN)==0: #Homo-deletion
                ValueToAdd=("1/1:"+str(round(float(BF),2))+":"+str(round(float(RR),2)))
                indexToKeep=listS.index(sample)+9 # the most 9 corresponds to the informative columns of the vcf .
                line[indexToKeep]=ValueToAdd
            else: #Hetero-deletion
                ValueToAdd=("0/1:"+str(round(float(BF),2))+":"+str(round(float(RR),2)))
                indexToKeep=listS.index(sample)+9
                line[indexToKeep]=ValueToAdd

    elif (line[4]=="<DUP>"): 
        ValueToAdd=("0/1:"+CN+":"+str(round(float(BF),2))+":"+str(round(float(RR),2)))
        indexToKeep=listS.index(sample)+9
        line[indexToKeep]=ValueToAdd

    return(line)


##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
###################################
def main(argv):
    ##########################################
    # A) ARGV parameters definition
    call = ''
    bf=''
    outputFile =''

    usage = """\nCOMMAND SUMMARY:
Given  a calling file of CNVs obtained by ExomeDepth.
Results are printed to stdout in vcf v4.3 format.
ARGUMENTS:
    --call [str]: A tsv file obtained in STEP2 (CNV calling results). 
                  Please indicate the full path.(15 columns)
    --bf [str]: Bayes factor filtering threshold (all CNVs with a 
                lower BF level will be removed from the analysis). 
    -out [str] : TODO COMPLETE \n"""
        
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","call=","bf=","out="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--call"):
            call=value     
        elif opt in ("--bf"):
            bf=value
        elif opt in ("--out"):
            outputFile=value
        else:
            sys.exit("ERROR : Programming error. Unhandled option "+opt+".\n")

    #####################################################
    # Checking that the mandatory parameters are presents
    if (call==""):
        sys.exit("ERROR : You must use --call.\n"+usage)

    if (bf==""):
        sys.exit("ERROR : You must use --bf.\n"+usage)

    if (outputFile==""):
        sys.exit("ERROR : You must use --out.\n"+usage)

    #####################################################
    # A) Input file format check
    tabToTreat=CNVCallSanityCheck(call)

    #####################################################
    # B)Filtering according to the Bayes Factor threshold chosen by the user in the function parameters.
    logger.info("Filtering")
    minBF=int(bf)
    CNVBFfilter=tabToTreat[tabToTreat.BF>minBF]
    logger.info("1) CNVs Number obtained after BF>=%s, filtering=%s/%s",minBF,len(CNVBFfilter),len(tabToTreat))

    #####################################################
    # C)Appropriate copy numbers allocation according to the read ratios.
    # Warning filtering of deletions with RR>0.75.
    CNVBFfilter.reset_index(drop=True, inplace=True)
    CNVBFfilter.loc[:,"CN"]=np.repeat(4,len(CNVBFfilter))#CN obsolete allow to delete DEL with RR>0.75
    CNVBFfilter.loc[CNVBFfilter["type"]=="duplication","CN"]=3
    CNVBFfilter.loc[CNVBFfilter["reads.ratio"]<0.75,"CN"]=1
    CNVBFfilter.loc[CNVBFfilter["reads.ratio"]<0.25,"CN"]=0
    CNVAllFilter=CNVBFfilter[CNVBFfilter["CN"]!=4]
    logger.info("2) CNVs Number obtained after copy number attribution  = %s/%s",len(CNVAllFilter),len(tabToTreat))

    #####################################################
    # D) Dictionnary Creation 
    #Key : CNV identifier: "CHR:start-end"
    #Value : list of patients characteristics affected by a cnv at this position. For each patient: SampleName_CN_BF_RR
    Dict={}
    for index, row in CNVAllFilter.iterrows():
        Key=row["id"]
        Value=row["sample"]+"_"+str(row["CN"])+"_"+str(row["BF"])+"_"+str(row["reads.ratio"])
        Dict.setdefault(Key,[]).append(Value)
    logger.info("3) Number of dictionnary Keys = %s",len(Dict))

    #####################################################
    # E) loop over the dictionary keys to create the final vcf.

    resultlist=[]
    sampleList=list(np.unique(CNVAllFilter["sample"]))
    for key, value in Dict.items():
        #######################################
        ## Key processing
        #chromosome treatment
        chrom=re.sub("^chr([\w]{1,2}):[\d]{3,9}-[\d]{3,9}$","\\1", key)
        #pos
        pos=re.sub("^chr[\w]{1,2}:([\d]{3,9})-[\d]{3,9}","\\1", key)
        pos=str(int(pos)+10)
        #end
        end=re.sub("^chr[\w]{1,2}:[\d]{3,9}-([\d]{4,9})","\\1", key)
        end=str(int(end)-10)

        #Empty lists initialization for identical positions between DUP and DEL within the cohort.
        firstline=[]
        secondline=[]

        #######################################
        ## Value processing
        for i in value:
            info_list=i.split("_")
            sample=info_list[0]          
            CN=info_list[1]
            BF=info_list[2]
            RR=info_list[3]          
            #First list initialisation           
            if len(firstline)==0:                
                firstline=VCFLine(sample,chrom,pos,key,end,CN,BF,RR,sampleList)  
            else:
                #add information to the first list
                if ((firstline[4]=="<DEL>") and (int(CN)<2)) or ((firstline[4]=="<DUP>") and (int(CN)>2)):
                    firstline=VCFLineCompleting(sample,CN,BF,RR,sampleList,firstline)
                #Second list initialisation      
                elif len(secondline)==0:
                    secondline=VCFLine(sample,chrom,pos,key,end,CN,BF,RR,sampleList)
                #add information to the second list
                elif len(secondline)>0:
                    secondline=VCFLineCompleting(sample,CN,BF,RR,sampleList,secondline)
                else:
                    logger.error("Unable to treat the CNV identified at the position:"+key+" for the patient:"+i+"."
                     +"\n Please check the correct format of the latter.")
                    sys.exit()

        #adding the results of the current key to the final list
        if len(secondline)>0:
            resultlist.append(firstline)
            resultlist.append(secondline)
        else:
            resultlist.append(firstline)   
            
    colNames=["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]+sampleList
    results=pd.DataFrame(resultlist, columns=colNames)# transform list to dataframe

    #####################################################
    # F) sorting
    results["CHR"]=results["#CHROM"]
    results["CHR"]=results["CHR"].str.replace('X', '23')
    results["CHR"]=results["CHR"].str.replace('Y', '24')
    results["CHR"]=results["CHR"].str.replace('M', '25')
    results["CHR"]=results["CHR"].astype(int)
    results["POS"]=results["POS"].astype(int)
    results=results.sort_values(by=["CHR","POS"])
    results=results.drop(columns=["CHR"])    

    #####################################################
    # G) Header definition
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

    #####################################################
    # H) Vcf saving
    output_VCF =os.path.join(outputFile,"CNVResults_BedtoolsExomeDepth_"+str(len(np.unique(CNVAllFilter["sample"])))+"samples_"+now+".vcf")
    with open(output_VCF, 'w') as vcf:
        vcf.write(header)

    results.to_csv(output_VCF, sep="\t", mode='a', index=False)

if __name__ =='__main__':
    main(sys.argv[1:])
    

