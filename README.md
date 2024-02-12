<h1 align="center"> Call CNVs from exome sequencing data </h1>

The pipeline enables germline Copy Number Variations (CNVs) to be called from human exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals associated with the canonical transcripts.<br>
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

### EXAMPLE USAGE:

##### STEP 1 : Fragments counting <br>

Given a BED of exons and one or more BAM files, count the number of sequenced fragments from each BAM that overlap each exon (+- padding).<br>
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after padding and sorting, subsequent columns (one per BAM) hold the counts.<br>
If a pre-existing counts file produced by this program with the same BED is provided (with --counts), counts for requested BAMs are copied from this file and counting is only performed for the new BAM(s).<br>
In addition, any support for putative breakpoints is printed to sample-specific TSV.gz files created in BPdir.<br>

Example:
```
BAMS="BAMs/sample1.bam,BAMs/sample2.bam"
BED="Transcripts_Data/ensemblCanonicalTranscripts.bed.gz"
python JACNEx/s1_countFrags.py --bams $BAMS --bed $BED --tmp /mnt/RamDisk/ --jobs 30 > fragCounts.tsv 2> step1.log
```

##### STEP 2 : Samples clustering <br>

Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million), performs quality control on the samples and forms the reference clusters for the call.<br>
The execution of the default command separates autosomes ("A") and gonosomes ("G") for clustering, to avoid bias (accepted sex chromosomes: X, Y, Z, W).<br>
Results are printed to stdout in TSV format: 5 columns [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]. <br>
In addition, all graphical support (quality control histogram for each sample and dendogram from clustering) are printed in pdf files created in plotDir.<br>

Example:
```
COUNT="fragCounts.tsv"
python JACNEx/s2_clusterSamps.py --counts $COUNT > resClustering.tsv 2> step2.log
```
##### STEP 3 : Copy numbers calls<br>
Accepts exon fragment count data (from 1_countFrags.py) and sample clustering information (from 2_clusterSamps.py) as input.<br>
Performs several critical operations:<br>
    a) Determines parameters for CN0 (half Gaussian) and CN2 (Gaussian) distributions for autosomal and gonosomal exons.<br>
    b) Excludes non-interpretable exons based on set criteria.<br>
    c) Calculates likelihoods for each CN state across exons and samples.<br>
    d) Generates a transition matrix for CN state changes.<br>
    e) Applies a Hidden Markov Model (HMM) to call and group CNVs.<br>
    f) Outputs the CNV calls in VCF format.<br>
The script utilizes multiprocessing for efficient computation and is structured to handle errors and exceptions effectively, providing clear error messages for troubleshooting.<br>
In addition, pie chart summarising exon filtering are produced as pdf files in plotDir.<br>

Example:
```
COUNT="fragCounts.tsv"
CLUST="clusters.tsv"
python JACNEx/s3_callCNVs.py --counts $COUNT --clusts $CLUST > callCNVs.vcf 2> step3.log
```

### CONFIGURATION:


### DEPENDENCIES:
It is necessary that all the software used are present. <br>
Samtools (tested with v1.15.1 - v1.18): <br>
```
wget https://github.com/samtools/samtools/releases/download/1.15.1/samtools-1.15.1.tar.bz2
tar -vxjf samtools-1.15.1.tar.bz2
cd samtools-1.15.1
./configure
make all all-htslib
```
It is also necessary to have python version >= 3.7 (3.6 and earlier have a bug that breaks JACNEx).

JACNEx also requires the following python modules:
numpy scipy numba ncls matplotlib scikit-learn KDEpy
We recommend the following commands, which cleanly install all the requirements in
a virtual environment, using the system-wide versions if available:
```
PYTHON=python3.11 ### or python3, or python, or... on ALMA9 we use python3.11
$PYTHON -m venv --system-site-packages ~/pyEnv_JACNEx
source ~/pyEnv_JACNEx/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib ncls numba scikit-learn KDEpy
```
On an ALMA9 system today (12/02/2024) this uses the system-wide:
numpy-1.23.5 (from python3.11-numpy-1.23.5-1.el9.x86_64)
SciPy-1.10.1 (from python3.11-scipy-1.10.1-2.el9.x86_64)

and it installs in ~/pyEnv_JACNEx/ :
matplotlib-3.8.2
ncls-0.0.68
numba-0.59.0
scikit-learn-1.4.0
KDEpy-1.1.8

