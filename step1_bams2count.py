



# -> a bed file, possibly gzipped but not padded or sorted (eg canonicalTranscripts_210826.bed.gz). Padding and sorting happens internally on-the-fly, this takes less than a second on fauve, see /home/nthierry/AmandineSeptier/Decona/Tests_countFrags_2201/README .

# -> optionally a countFile.tsv with counts from previously processed BAMs, this countFile is as described above and similar to AS's current Readcount_BindBedtoolsRes_2022-01-11.tsv .
# If this file is provided:
# - check that the first 4 columns are identical to the provided BED file (after calling processBed), if not die with a warning;
# - identify all samples that are already counted in the file.

# -> a tmpDir (for samtools sort), defaults to /tmp

# -> a list of one or more bamFiles to process, ASSUMPTION: each bamfile is called [$sample].bam and we use this to grab sample identifiers and compare them to the sampleIDs already counted in countFile

import sys
import os
import logging
import pandas as pd # dataFrame in processBed
import re



#############################################################################
### subs
#############################################################################

#################################################
# processBed:
# - bedFile == a bed file (with path), possibly gzipped, containing exon definitions
#   formatted as CHR START END EXON_ID
#
# Returns the data as a pandas dataframe with column headers CHR START END EXON_ID,
# and where:
# - a 10bp padding is added to exon coordinates (ie -10 for START and +10 for END)
# - exons are sorted by CHR, then START, then END, then EXON_ID
def processBed(bedFile, logger):
    try:
        exons=pd.read_table(bedFile,sep="\t", header=None,
                            names=['CHR', 'START', 'END', 'EXON_ID'],
                            dtype= {'CHR':str, 'START':int, 'END':int, 'EXON_ID':str})
        # compression == 'infer' by default => auto-works whether bedFile is gzipped or not
    except Exception as e:
        logger.error("error parsing bedFile %s: %s", bedFile, e)
        sys.exit(1)
    # NOTES: this dies as expected if start or end is missing or not a number,
    # but a missing chr or exon_id is just populated as NaN and doesn't kill.
    # Also, if first row has 5 fields we die as expected, but any 5th (or more)
    # field in any subsequent row is simply ignored... This is in contradiction
    # with the pandas.read_table doc which states that lines "with too many fields
    # raise an exception", but we'll just have to hope the bedFile is OK.

    # pad coordinates with +-10bp
    exons['START'] -= 10
    exons['END'] += 10

    # add temp CHR_NUM column so we can sort on CHR:
    exons['CHR_NUM'] = exons['CHR']
    # remove leading 'chr' if present
    exons['CHR_NUM'].replace(regex=r'^chr(\w+)$', value=r'\1', inplace=True)
    # replace X Y M by 97 98 99 (if present)
    exons.loc[exons['CHR_NUM']=='X','CHR_NUM'] = 97
    exons.loc[exons['CHR_NUM']=='Y','CHR_NUM'] = 98
    exons.loc[exons['CHR_NUM']=='M','CHR_NUM'] = 99
    # convert type of CHR_NUM to int and catch any errors
    try:
        exons['CHR_NUM'] = exons['CHR_NUM'].astype(int)
    except Exception as e:
        logger.error("error converting CHR_NUM to int: %s", e)
        sys.exit(1)
    # sort by CHR_NUM, then START, then END, then EXON_ID
    exons = exons.sort_values(by=['CHR_NUM','START','END','EXON_ID'])
    # delete the temp column, and return result
    exons.drop(['CHR_NUM'], axis=1, inplace=True)
    return(exons)


#################################################
# parseCountFile:
# - countFile is a tsv file (with path), including column titles, as
#   specified previously
# - exons is a dataframe holding exon definitions, padded and sorted,
#   as returned by processBed
#
# -> Parse countFile into a dataframe (will be returned)
# -> Check that the first 4 columns are identical to exons,
#    otherwise die with an error.
def parseCountFile(countFile, exons, logger):
    try:
        counts=pd.read_table(countFile,sep="\t")
    except Exception as e:
        logger.error("error parsing provided countFile %s: %s", countFile, e)
        sys.exit(1)

    


#############################################################################
### main
#############################################################################


def main(argv):
    ###################################
    # set up logger
    logger=logging.getLogger(os.path.basename(sys.argv[0]))
    logger.setLevel(logging.DEBUG)
    # create console handler for STDERR
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.DEBUG)
    #create formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    #add formatter to stderr handler
    stderr.setFormatter(formatter)
    #add stderr handler to logger
    logger.addHandler(stderr)

    ###################################
    # USAGE
    
    ###################################
    # arguments: declare with defaults if any, parse+check and populate with getopt
    bedFile = 'canonicalTranscripts_210826.bed.gz'
    #bedFile = 'canonicalTranscripts_210826_PADDED_SORTED.bed'
    #bedFile = 'canonicalTranscripts_210826_PADDED_SORTED_BAD.bed'

    # check provided args
    if not os.path.isfile(bedFile):
        logger.error("bedFile %s doesn't exist", bedFile)
        sys.exit()

    ###################################
    # parse Bed
    logger.info("looking good, start by parsing bedFile %s", bedFile)
    exons = processBed(bedFile, logger)

    # DEBUG: print to TSV
    exons.to_csv("exons_padded_sorted.bed",index=False, header=False,sep="\t")
    # DEBUG: compare with AS's version STEP0_GRCH38_* -> very similar except
    # when 2 exons have identical coords I sort by transcript while AS returns
    # the exons in random order (sort_values is not stable by default)

    
    # if countFile was provided: parse it and check that it was built with bedFile
    



### main
if __name__ =='__main__':
    main(sys.argv[1:])
