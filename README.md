<h1 align="center"> CNV calls for exome sequencing data from human cohort </h1>


The pipeline enables germline Copy Number Variations (CNVs) to be called from human exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals associated with the canonical transcripts.<br>
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

### EXAMPLE USAGE:

* STEP 0 : Interval bed creation <br>

This step consists in creating an interval file in the bed format.<br>
It performs a verification of the input bed file, performs a padding +-10pb and sorts the data in genomic order.<br>
It is necessary to respect the format of the reference genome name for pipeline interoperability at each new process started.<br>
```
BED="canonicalTranscripts.bed.gz"
GENOMEV="GRCH38_vXXX"
OUTPUT="~/Scripts/"
/bin/time -v ~/FolderContainsScript/python3.6 STEP_0_IntervalList.py -b $BED -n $GENOMEV -o $OUTPUT 2> ./.err
```

* STEP 1 : Counting reads <br>

This step uses the bam files to record the number of reads overlapping the intervals present in the bed file created in step 0.<br>
It will create a new folder for storing the results. <br>
This script uses Bedtoolsv2.29.1 and its multibamCov program.<br>
It is executed in parallel to reduce the process time.<br>
Warning: the number of cpu used for parallelization is optimized.<br>
If the number of cpu is increased manually within the script it can lead to waiting times increasing the global process time.<br>
The script provides a read count file for each sample analyzed (tsv files).<br>

```
INTERVAL="~/STEP0_GRCH38_vXXX_Padding10pb_NBexons_Date.bed"
BAM="~/BAMs/"
OUTPUT="~/SelectOutputFolder/"
/bin/time -v ~/FolderContainsScript/python3.6 STEP_1_CollectReadCounts_Bedtools.py -i $INTERVAL -b $BAM -o $OUTPUT 2> ./.err 
```

* STEP 2 : CNV Calling<br>

This step performs the CNV calling.<br>
However it uses the DECON/ExomeDepth script in R modified to allow inserting different inputs (tsv instead of Rdata).<br>
It has also been modified by adding a sanity check of the input files.<br>
The R script does not have the part allowing to generate the plots anymore.<br>
The output file is in tsv format and contains the complete CNV calling results for all samples.<br>
```
INTERVAL="~/STEP0_GRCH38_v104_Padding10pb_NBexons_Date.bed"
READF="~/Bedtools/"
OUTPUT="~/SelectOutputFolder/"
/bin/time -v ~/FolderContainsScript/python3.6 STEP_2_GermlineCNVCaller_DECoNBedtools.py -i $BED -r $READF -o $OUTPUT 2> ./.err
```

* STEP 3 : VCF Formatting<br>

This step allows the conversion of CNV calling results into vcfv4.3 format that can be interpreted by the VEP software (annotation of structural variants) <br>
The Bayes Factor(BF) parameter is user selectable. It will be necessary to perform a first filtering of the CNV. <br>
```
CALL="~/Results_BedtoolsCallingCNVExomeDepth_Date.tsv"
BF="20"
OUTPUT="~/SelectOutputFolder/Calling_results_Bedtools_Date/"
/bin/time -v ~/FolderContainsScript/python3.6 STEP_3_VCFFormatting_Bedtools.py -c $CALL -b $BF -o $OUTPUT 2> ./.err
```

### CONFIGURATION:
To launch the different stages of the pipeline it is necessary to be located in the folder where you want to obtain the outputs. <br>

### DEPENDENCIES:
It is necessary that all the software used is present. <br>
DECON: <br>
```
git clone https://github.com/RahmanTeam/DECoN
```
Bedtools 2:<br>
```
wget https://github.com/arq5x/bedtools2/releases/download/v2.29.1/bedtools-2.29.1.tar.gz
tar -zxvf bedtools-2.29.1.tar.gz
cd bedtools2
make
```


