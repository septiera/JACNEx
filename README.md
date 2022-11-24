<h1 align="center"> CNV calls for exome sequencing data from human cohort </h1>

The pipeline enables germline Copy Number Variations (CNVs) to be called from human exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals associated with the canonical transcripts.<br>
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

### EXAMPLE USAGE:

## STEP 1 : Fragments counting <br>

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
python 1_CountFrags.py --bams-from $BAM --bed $BED --counts $COUNT --tmp $TMP --threads 20 --jobs 3 > $OUT 2> $ERR

```

# STEP 2 : Samples clustering <br>

Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and forms the reference clusters for the call. <br>
By default, separation of autosomes ("A") and gonosomes ("G") for clustering, to avoid bias (chr accepted: X, Y, Z, W).<br>
Results are printed to stdout folder:<br>
- a TSV file format, describe the clustering results, dim = NbSOIs*8 columns:<br>
    1) "sampleID": name of interest samples [str],<br>
    2) "clusterID_A": clusters identifiers [int] obtained through the normalized fragment counts of exons on autosomes, <br>
    3) "controlledBy_A": clusters identifiers controlling the sample cluster [str], a comma-separated string of int values (e.g "1,2"). If not controlled empty string.<br>
    4) "validitySamps_A": a boolean specifying if a sample is dubious(0) or not(1)[int]. This score set to 0 in the case the cluster formed is validated and does not have a sufficient number of individuals.<br>
    5) "genderPreds": a string "M"=Male or "F"=Female deduced by kmeans,<br>
The columns 6, 7 and 8 are the same as 2, 3 and 4 but are specific to gonosomes.<br>
In case the user doesn't want to discriminate genders, the output tsv will contain the format of the first 4 columns for all chromosomes.<br>
- one or more png's illustrating the clustering performed by dendograms. [optionnal]<br>
    Legend : solid line = control clusters , thin line = target clusters<br>
    The clusters appear in decreasing order of distance (1-|pearson correlation|).<br>

```
COUNT="FragCount.tsv"
OUT="ResultFolder"
ERR="step2.err"
python 2_clusterSamps.py --counts $COUNT --out $OUT --figure 2> $ERR
```
# STEP 3 : Copy numbers calls<br>

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
