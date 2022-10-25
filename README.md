<h1 align="center"> CNV calls for exome sequencing data from human cohort </h1>

The pipeline enables germline Copy Number Variations (CNVs) to be called from human exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals associated with the canonical transcripts.<br>
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

### EXAMPLE USAGE:

* STEP 1 : Counting step  <br>

Given a BED of exons and one or more BAM files, count the number of sequenced fragments from each BAM that overlap each exon (+- padding).<br>
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after padding and sorting, subsequent columns (one per BAM) hold the counts.<br>
If a pre-existing counts file produced by this program with the same BED is provided (with --counts), counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).<br>
This script uses Samtoolsv1.15.1.<br>

```

BAM="sample1.bam,sample2.bam"
BED="EnsemblCanonicalTranscripts.bed.gz"
TMP="/mnt/RamDisk/"
OUT="FragCount.tsv"
ERR="step1.err"
python STEP_1_CollectReadCounts_MAGE_CNV.py --bams-from $BAM --bed $BED --counts $COUNT --tmp $TMP --threads 20 --jobs 3 > $OUT 2> $ERR

```

* STEP 2 : Select Sample Group <br>

Given a TSV of exon fragment counts and a TSV of sample gender information, normalizes the counts (Fragment Per Million) and forms the reference groups for the call.<br>
Results are printed to stdout folder : <br>
- a TSV file format: first 4 columns hold the exon definitions, subsequent columns hold the normalised counts.<br>
- a TSV file format: describe the distribution of samples in the reference groups (7 columns); first column sample of interest (SOIs), second reference group number for autosomes, third minimum group correlation level for autosomes, fourth sample valid status, fifth sixth and seventh identical but for sex chromosome.<br>

```
COUNT="FragCount.tsv"
META="CohortInfo.csv"
OUT="ResultFolder"
ERR="step2.err"
python STEP_2_CountNormalisation_SelectReferenceGroup_MAGE_CNV.py --counts $COUNT --metadata $META --out $OUT 2> $ERR
```

* STEP 3 : CNV Calling<br>

### CONFIGURATION:
To launch the different stages of the pipeline it is necessary to be located in the folder where you want to obtain the outputs. <br>

### DEPENDENCIES:
It is necessary that all the software used are present. <br>
Samtools (v1.15.1): <br>
```
wget https://github.com/samtools/samtools/releases/download/1.15.1/samtools-1.15.1.tar.bz2
tar -vxjf samtools-1.15.1.tar.bz2
cd samtools-1.15.1
./configure
make all all-htslib
```
It is also necessary to have python version 3.6.
As well as the following modules:
```
python3 -m venv ~/pythonEnv36
pip3 install numpy scipy

numpy v1.19.5
scipy v1.5.4

```
