<h1 align="center"> Call CNVs from exome sequencing data </h1>

The pipeline enables germline Copy Number Variations (CNVs) to be called from human exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals associated with the canonical transcripts.<br>
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

### EXAMPLE USAGE:

##### STEP 1 : Fragments counting <br>

Given a BED of exons and one or more BAM files, count the number of sequenced fragments from each BAM that overlap each exon (+- padding).<br>
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after padding and sorting, subsequent columns (one per BAM) hold the counts.<br>
If a pre-existing counts file produced by this program with the same BED is provided (with --counts), counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).<br>
In addition, any support for putative breakpoints is printed to sample-specific TSV files created in BPdir.<br>

Example:
```
BAMS="BAMs/sample1.bam,BAMs/sample2.bam"
BED="Transcripts_Data/ensemblCanonicalTranscripts.bed.gz"
python MAGe_CNV/1_countFrags.py --bams $BAMS --bed $BED --tmp /mnt/RamDisk/ --jobs 30 > fragCounts.tsv 2> step1.log
```

##### STEP 2 : Samples clustering <br>

Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million), performs quality control on the samples and forms the reference clusters for the call.<br>
The execution of the default command separates autosomes ("A") and gonosomes ("G") for clustering, to avoid bias (accepted sex chromosomes: X, Y, Z, W).<br>
Results are printed to stdout in TSV format: 5 columns [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]. <br>
In addition, all graphical support (quality control histogram for each sample and dendogram from clustering) are printed in pdf files created in plotDir.<br>

Example:
```
COUNT="fragCounts.tsv"
python MAGe_CNV/2_clusterSamps.py --counts $COUNT > resClustering.tsv 2> step2.log
```
##### STEP 3 : Copy numbers calls<br>

### CONFIGURATION:
To launch the different stages of the pipeline it is necessary to be located in the folder where you want to obtain the outputs. <br>

### DEPENDENCIES:
It is necessary that all the software used are present. <br>
Samtools (tested with v1.15.1): <br>
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
python3 -m venv ~/pyEnv_MageCNV
source pyEnv_MageCNV/bin/activate
pip install --upgrade pip
pip install numpy scipy numba ncls matplotlib scikit-learn

numpy v1.19.5
scipy v1.5.4
matplotlib v3.3.4
scikit-learn v0.24.2


```
