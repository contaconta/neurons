clear all; close all; clc;
%%
addpath(genpath('../'));
%%
run('../DetectionEvaluation/prerequisites');
if 0
    matlabpool local; %#ok
end
overlappingTolerance = 0.5;
isDetectionDone = false;
extractSamples  = true;
%%
Magnification = '10x';
dataRootDirectory    = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic/'];
ConvertedGTRootDir   = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic_matlab/'];
RawRootDataDirectory = ['/Users/feth/Documents/Work/Data/Synergia/Selection' Magnification '/'];
DetectionDirectory   = ['/Users/feth/Documents/Work/Data/Synergia/Detections' Magnification '/'];
if(~exist(DetectionDirectory, 'dir'))
    mkdir(DetectionDirectory);
end
%% first, detect cell bodies and save them if needed
if ~isDetectionDone
    PreprocessAndSaveCellBodyDetections(Magnification, dataRootDirectory, RawRootDataDirectory, DetectionDirectory);
end
%% Given the ground truth and the detection, Train
FastEMD_parameters.NUMBER_OF_BINS      = 32;
FastEMD_parameters.THRESHOLD_BINS_DIST = 5;
save('FastEMDParams', 'FastEMD_parameters');

load('FastEMDParams');

if extractSamples
   [PositiveEMDs, NegativeEMDs] = ExtractEMDSamples(Magnification, dataRootDirectory, ConvertedGTRootDir, DetectionDirectory, overlappingTolerance, FastEMD_parameters);
end
%% Logistic regression
B = SigmoidFitting(PositiveEMDs, NegativeEMDs);
FileNameSigmoidParams = ['SigmoidParams' Magnification];
save(FileNameSigmoidParams, 'B');
%%
listOfNegToUse = randi(numel(NegativeEMDs), numel(PositiveEMDs), 1);
X = [PositiveEMDs; NegativeEMDs(listOfNegToUse)];
Z = Logistic(B(1) + X * (B(2)));
%%
delta = 0.1;
XX = 0:delta:4;
ZZ = Logistic(B(1) + XX * (B(2)));
figure; plot(XX, ZZ)

