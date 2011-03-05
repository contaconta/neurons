#!/bin/bash


# For the detections - Area is different
for i in MeanRedIntensity Speed MeanGreenIntensity Eccentricity TravelDistance deltaArea deltaEccentricity Perimeter deltaMeanGreenIntensity deltaPerimeter Acc; do

sed s/AreaNuclei/${i}Nuclei/ < ffAreaNuclei.m  > tmp
sed s/Area/${i}/ < tmp  > ff${i}Nuclei.m
rm -rf tmp
done