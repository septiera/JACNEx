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

#### samtools
JACNEx needs samtools (tested with v1.15.1 - v1.18), can be installed with: <br>
```
wget https://github.com/samtools/samtools/releases/download/1.18/samtools-1.18.tar.bz2
tar xfvj samtools-1.18.tar.bz2
cd samtools-1.18
./configure
make all all-htslib
```
You then need to place samtools-1.18/samtools in your $PATH (e.g. create a symlink to it in /usr/local/bin/ if you are sudoer), or pass it to JACNEx.py with --samtools= .

#### python
JACNEx needs python version >= 3.7 (3.6 and earlier have a bug that breaks JACNEx).
For example on ALMA Linux 9 we use python3.12, available in the standard repos since ALMA 9.4:
```
sudo dnf install python3.12 python3.12-setuptools python3.12-numpy python3.12-scipy
sudo dnf install python3.12-pip-wheel python3.12-setuptools-wheel python3.12-wheel-wheel
```

#### python modules
JACNEx requires the following python modules:<br>
_numpy scipy numba ncls matplotlib pyerf scikit-learn KDEpy_<br>
We recommend the following commands, which cleanly install all the requirements in
a python virtual environment, using the system-wide versions if available:
```
PYTHON=python3.12 ### or python3, or python, or...
$PYTHON -m venv --system-site-packages ~/pyEnv_JACNEx
source ~/pyEnv_JACNEx/bin/activate
pip install --upgrade pip
pip install numpy scipy numba ncls matplotlib pyerf scikit-learn KDEpy
```
On an ALMA9.4 system today (28/05/2024) this uses the system-wide:<br>
**numpy-1.24.4 scipy-1.11.1**

and it installs in ~/pyEnv_JACNEx/ :<br>
**numba-0.59.1 ncls-0.0.68 matplotlib-3.8.4 pyerf-1.0.1 scikit_learn-1.4.2 KDEpy-1.1.9**

You then need to activate the venv before running JACNEx, e.g.:
```
$ source ~/pyEnv_JACNEx/bin/activate
(pyEnv_JACNEx) $ python path/to/JACNEx/JACNEx.py --help
```
