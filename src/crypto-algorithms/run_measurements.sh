#! /bin/bash

set -o xtrace

MEASUREMENTS=10
NAMES=('arcfour' 'des' 'rot-13')
FILE_NAMES=('sample_files/moby_dick.txt' 'sample_files/king_james_bible.txt' 'sample_files/hubble_1.tif' 'sample_files/mercury.png')

make
mkdir results

for NAME in ${NAMES[@]}; do
    for FILE in ${FILE_NAMES[@]}; do
        perf stat -r $MEASUREMENTS ./$NAME '-tf' $FILE  >> $NAME.log 2>&1
    done
    mv *.log results/
done
