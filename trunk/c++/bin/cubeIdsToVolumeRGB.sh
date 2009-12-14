#!/bin/bash

outName=$2

# idsToVolume $1 $outName

for i in R G B; do
inputName=`printf %s%s.nfo $outName $i`
outNameC=`printf %s%su $outName $i`
# cubeConvertToUchar $inputName $outNameC
echo "filenameVoxelData$i $outNameC.nfo" >> $outName.nfc
done
echo 'type color' >> $outName.nfc

