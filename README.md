<h1 align="center"> CNV calls for exome sequencing data from human cohort </h1>


The pipeline enables germline Copy Number Variations (CNVs) to be called from human exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals<br>
associated with the canonical transcripts.
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

### EXAMPLE USAGE:

* STEP 1 : Counting step  <br>
 * Bedtools (reads counting)<br>

This step uses the bam files to record the reads number overlapping the intervals present in a bed file (not padded).<br>
This script uses Bedtoolsv2.26.0 and its multicov program.<br>
The script provides a read count file for each sample analyzed (tsv files).<br>

```
BAM="sample1.bam,sample2.bam"
BED="EnsemblCanonicalTranscripts.bed.gz"
OUT="FragCount.tsv"
ERR="step1.err"
# optionnal parameters
COUNT="OldFragCount.tsv"
# execution
python3 STEP_1_CollectReadCounts_Bedtools.py --bams $BAM --bed $BED --counts $COUNT > $OUT 2> $ERR 

```

 * Customized fragments counting

This script allows to perform a fragment count. <br>
It also uses bam as input, and samtools v1.9 (using htslib 1.9) to sort and filter them.
The bed used is padded by +-10bp (to possibly capture uncovered exons, but catchable by fragment counting)
The outputs of this script are identical to those of bedtools, i.e. tsv containing 5 first columns (CHR, START, END, EXON_ID),<br>
following columns are the count results for each sample.<br>
The files are thus directly usable by the next step.<br>
It is possible to complete the counts when adding new patients.<br>

```
# mandatory parameters
BAM="sample1.bam,sample2.bam"
BED="EnsemblCanonicalTranscripts.bed.gz"
TMP="/tmp/"
OUT="FragCount.tsv"
ERR="step1.err"
# optionnal parameters
COUNT="OldFragCount.tsv"
# execution
python3 STEP_1_CollectReadCounts_DECONA2.py --bams $BAM --bed $BED --counts $COUNT --tmp $TMP --threads 20 > $OUT 2> $ERR 

```

* STEP 2 : CNV Calling<br>

This step performs the CNV calling.<br>
However it uses the ExomeDepth(v1.1.15) script in R .<br>
The calling is based on the betabinomial and the segmentation on the HMM (hidden markov model)<br>
The input file is .tsv counting reads(bedtools) or fragments(customized counting).<br>
There are two output files: one for the calling results and the other for keeping track of the reference sets used in the process.<br>
In this script there is also the calling check when a new sample is added to the cohort.<br>

```
# mandatory parameters
COUNT="FragCount.tsv"
OUTC="CallingResults_ExomeDepth.tsv"
OUTR="RefSet_ExomeDepth.tsv"
ERR="step2_CNVcall.err"
# optionnal parameters
OLDC="OldCallingResults_ExomeDepth.tsv"
OLDR="OldRefSet_ExomeDepth.tsv"
# execution
python3 STEP_2_GermlineCNVCaller_DECoNBedtools.py --counts $COUNT --outputcalls $OUTC --outputrefs $OUTR --calls $OLDC --refs $OLDR 2> $ERR

```

* STEP 3 : VCF Formatting<br>

This step allows the conversion of CNV calling results into vcf 4.3 format that can be interpreted by the VEP software (annotation of structural variants) <br>
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
It is necessary that all the software used are present. <br>
DECON (v1.0.2): <br>
```
git clone https://github.com/RahmanTeam/DECoN
```
ExomeDepth (v1.1.15):
```
git clone https://github.com/vplagnol/ExomeDepth
```
Bedtools 2:<br>
```
wget https://github.com/arq5x/bedtools2/releases/download/v2.26.0/bedtools-2.26.0.tar.gz
tar -zxvf bedtools-2.26.0.tar.gz
cd bedtools2
make
```
It is also necessary to have python version 3.6.
As well as the following modules:
```
python3 -m venv ~/pythonEnv36
pip3 install ncls pandas notebook

pandas version 1.1.5
ncls version 0.0.63
```
