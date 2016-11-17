#!/bin/bash

HG38_RMSK="http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz"
CLASS="LINE SINE"

set -e -o pipefail

if [ ! -f ./rmsk.txt.gz ]; then
    wget $HG38_RMSK
fi

for class in $CLASS; do
    echo "Creating bed-file for $class..."
    zcat rmsk.txt.gz | \
    awk -v class=$class \
    'BEGIN{OFS="\t"}{if($12==class) {print($6, $7, $8, $11 ":" $12 ":" $13);}}' > rmsk_$class.bed
done

