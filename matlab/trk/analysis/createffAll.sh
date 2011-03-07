#!/bin/bash


# For the detections in the nucleus
for i in MeanRedIntensity Speed MeanGreenIntensity Eccentricity TravelDistance deltaArea deltaEccentricity Perimeter deltaMeanGreenIntensity deltaPerimeter Acc; do
sed s/AreaNuclei/${i}Nuclei/ < ffAreaNuclei.m  > tmp
sed s/Area/${i}/ < tmp  > ff${i}Nuclei.m
rm -rf tmp
done


# For the detections in the soma

for i in  Eccentricity Perimeter MeanGreenIntensity deltaArea deltaPerimeter deltaMeanGreenIntensity deltaEccentricity Speed Acc TravelDistance; do
sed s/AreaSomata/${i}Somata/ < ffAreaSomata.m  > tmp
sed s/Area/${i}/ < tmp  > ff${i}Somata.m
rm -rf tmp
done

for i in NucleusTimeContracting  NucleusFreqExpansion SomaTimeExpanding  SomaTimeContracting GermanTotalCableLengthTimeExpanding GermanTotalCableLengthTimeContracting GermanTotalCableLengthFreqExpansion NumTrackedNeuritesTimeExpanding NumTrackedNeuritesTimeContracting NumTrackedNeuritesFreqExpansion GermanNumNeuritesTimeExpanding GermanNumNeuritesTimeContracting GermanNumNeuritesFreqExpansion GermanTotalCableLengthFilopodiaTimeExpanding GermanTotalCableLengthFilopodiaTimeContracting GermanTotalCableLengthFilopodiaFreqExpansion; do

sed s/NucleusTimeExpanding/${i}/ < fnNucleusTimeExpanding.m > fn${i}.m

done