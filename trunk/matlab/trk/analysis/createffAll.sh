#!/bin/bash


## INFORMATION ON THE DETECTIONS (preffix d)
# For the detections in the nucleus - D
for i in MajorAxisLength MinorAxisLength Eccentricity Orientation Perimeter MeanGreenIntensity MeanRedIntensity deltaArea deltaPerimeter deltaMeanGreenIntensity deltaEccentricity Speed Acc TravelDistance KevinTotalCableLengthExpand; do
sed s/AreaNuclei/${i}Nuclei/ < ffAreaNuclei.m  > tmp
sed s/Area/${i}/ < tmp  > ff${i}Nuclei.m
rm -rf tmp
done

# For the detections in the soma - N
for i in  MajorAxisLength MinorAxisLength Eccentricity Perimeter MeanGreenIntensity deltaArea deltaPerimeter deltaMeanGreenIntensity deltaEccentricity Speed Acc TravelDistance Expand; do
sed s/AreaSomata/${i}Somata/ < ffAreaSomata.m  > tmp
sed s/Area/${i}/ < tmp  > ff${i}Somata.m
rm -rf tmp
done

# Detections in the neurites per frame - Soma
for i in DistToSomaExtreme DistToSomaMean DistToSomaMedian DistToSomaStandDev Eccentricity FiloCableLength FiloCount FiloPercent MajorAxisLength MinorAxisLength RadialDotProd TotalCableLength Expand deltaBranchCount deltaDistToSomaExtreme deltaDistToSomaStandDev deltaEccentricity deltaFiloCableLength deltaFiloCount deltaFiloPercent deltaMajorAxisLength deltaRadialDotProd deltaTotalCableLength; do
sed s/BranchCount/${i}/ < ffBranchCountNeurite.m > ff${i}Neurite.m
done


##INFORMATION ON THE TRACKS OF THE NEURITES (preffix a) -[ N(trkSeq{d})]
for i in BranchCount DistToSomaMean DistToSomaMedian DistToSomaStandDev Eccentricity FiloCableLength FiloCount FiloPercent MajorAxisLength MinorAxisLength RadialDotProd TotalCableLength; do
sed s/DistToSomaExtreme/${i}/ < faMaxDistToSomaExtremeNeurite.m > faMax${i}Neurite.m
sed s/DistToSomaExtreme/${i}/ < faMeanDistToSomaExtremeNeurite.m > faMean${i}Neurite.m
done

# Information pre-computed on neurites - NTimeInfo
for i in  BranchCountContracting BranchCountFreqExpansion DistToSomaExtremeExpanding DistToSomaExtremeContracting DistToSomaExtremeFreqExpansion MajorAxisLengthExpanding MajorAxisLengthContracting MajorAxisLengthFreqExpansion EccentricityExpanding EccentricityContracting EccentricityFreqExpansion FiloCountExpanding FiloCountContracting FiloCountFreqExpansion FiloCableLengthExpanding FiloCableLengthContracting FiloCableLengthFreqExpansion TotalCableLengthExpanding TotalCableLengthContracting TotalCableLengthFreqExpansion; do
sed s/BranchCountExpanding/${i}/ < faBranchCountExpanding.m > fa${i}.m
done







## INFORMATION ON THE TRACKS OF THE NEURONS (preffix n)
# Temporal Neuron Information
for i in NucleusTimeExpanding NucleusTimeContracting NucleusFreqExpansion SomaTimeExpanding SomaTimeContracting SomaFreqExpansion GermanTotalCableLengthTimeExpanding GermanTotalCableLengthTimeContracting GermanTotalCableLengthFreqExpansion NumTrackedNeuritesTimeExpanding NumTrackedNeuritesTimeContracting NumTrackedNeuritesFreqExpansion GermanNumNeuritesTimeExpanding GermanNumNeuritesTimeContracting GermanNumNeuritesFreqExpansion GermanTotalCableLengthFilopodiaTimeExpanding GermanTotalCableLengthFilopodiaTimeContracting GermanTotalCableLengthFilopodiaFreqExpansion KevinTotalCableLengthTimeExpanding KevinTotalCableLengthTimeContracting KevinTotalCableLengthFreqExpansion KevinTotalCableLengthFilopodiaTimeExpanding KevinTotalCableLengthFilopodiaTimeContracting KevinTotalCableLengthFilopodiaFreqExpansion KevinBranchCountTimeExpanding KevinBranchCountTimeContracting KevinBranchCountFreqExpansion KevinFiloCountTimeExpanding KevinFiloCountTimeContracting KevinFiloCountFreqExpansion ; do

sed s/NucleusTimeExpanding/${i}/ < fnNucleusTimeExpanding.m > fn${i}.m
done


for i in KevinTotalCableLength KevinTotalCableLengthFilopodia KevinBranchCount KevinFiloCount NumTrackedNeurites GermanNumNeurites GermanTotalCableLengthFilopodia; do

sed s/GermanTotalCableLength/${i}/ < fnMeanGermanTotalCableLength.m > fnMean${i}.m
sed s/GermanTotalCableLength/${i}/ < fnSTDGermanTotalCableLength.m > fnSTD${i}.m
sed s/GermanTotalCableLength/${i}/ < fnMaxGermanTotalCableLength.m > fnMax${i}.m
done



