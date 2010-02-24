#!/bin/bash

#videoSteer imageFolder vector outputImage

EARGS=3
if [ $# -ne $EARGS ]
then
    echo "usage:\nvideoSteer imageFolder vector outputImage\n"
    exit 1;
fi

dirOrig=$1
vector=$2
outputImage=$3
sigma=1

printf "directory = %s, vector = %s, outputImage = %s\n" $1 $2 $3

for i in delta gaussian radius toroid linear linearInv squared random sine sine2; do
    dirImage=`printf %s/%s $dirOrig $i`
    nameImage=`printf %s/%s.png $dirImage $i`
    echo $nameImage
    for angle in `seq 0 30 150`; do
        steer $nameImage $vector $sigma $angle
        nameOutImage=`printf %s/result_%03i.jpg $dirImage $angle`
        mv $dirImage/result.jpg  $nameOutImage
    done
    resultList=`ls $dirImage/result_*`
    convert $nameImage $resultList +append $dirImage/row.png
done

rows=`find $dirOrig -name row.png`
convert $rows -append $3