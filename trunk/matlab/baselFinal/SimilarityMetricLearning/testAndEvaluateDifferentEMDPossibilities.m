clear all; close all; %clc;
%%
addpath(genpath('../'));
%%
run('../DetectionEvaluation/prerequisites');
if 0
    matlabpool local; %#ok
end
overlappingTolerance = 0.5;
isDetectionDone = true;
%%
Magnification = '10x';
dataRootDirectory    = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic/'];
ConvertedGTRootDir   = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic_matlab/'];
RawRootDataDirectory = ['/Users/feth/Documents/Work/Data/Sinergia/Olivier/Selection' Magnification '/'];
DetectionDirectory   = ['/Users/feth/Documents/Work/Data/Sinergia/Olivier/Detections' Magnification '/'];
if(~exist(DetectionDirectory, 'dir'))
    mkdir(DetectionDirectory);
end
%%
listOfGTSeq = dir(dataRootDirectory);
AllTruePositives = [];
disp('========================================')
seqIdx = 1;



for i = 1:length(listOfGTSeq)
   if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
       inputSeqDirToProcess = [RawRootDataDirectory listOfGTSeq(i).name '/'];
      
       detectionsFileName = [DetectionDirectory listOfGTSeq(i).name];
       if ~isDetectionDone
           [Nuclei, Somata, Cells, CellsList] =  trkDetectNucleiSomataForEvaluation(inputSeqDirToProcess, Magnification);
           save(detectionsFileName, 'Cells', 'CellsList');
       else
           load(detectionsFileName);
       end
       
       CellsPerSeq{seqIdx} = Cells;
       CellsListPerSeq{seqIdx} = CellsList;
       
       GTFileName = [ConvertedGTRootDir listOfGTSeq(i).name];       
       load(GTFileName);
       
       [TruePositives, FalsePositives, truePosTrkId, truePosTime] = trkEvaluateDetections(Cells, CellsList, AnnotatedTrackedCells, overlappingTolerance);
       PositivePairs = [];
       NegativePairs = [];
       for k =1:max(truePosTime)-1
           % find the pair of indices such that 
           StartIdx         = find(truePosTime ==  k );
           EndIdx           = find(truePosTime == k+1);
           [p, q]           = meshgrid(StartIdx, EndIdx);
           pairsIdx         = [p(:) q(:)];
           PositivePairsIdx = pairsIdx(truePosTrkId(pairsIdx(:, 1)) == truePosTrkId(pairsIdx(:, 2)), :);
           NegativePairsIdx = pairsIdx(truePosTrkId(pairsIdx(:, 1)) ~= truePosTrkId(pairsIdx(:, 2)), :);
           PositivePairs    = [PositivePairs; TruePositives(PositivePairsIdx)];%#ok
           NegativePairs    = [NegativePairs; TruePositives(NegativePairsIdx)];%#ok
       end
       
       PositivePairsPerSeq{seqIdx} = Cells(PositivePairs);%#ok
       NegativePairsPerSeq{seqIdx} = Cells(NegativePairs);%#ok

       FalsePositivesPerImage{seqIdx}  = FalsePositives;%#ok
       TruePositivesPerImage{seqIdx}   = TruePositives;%#ok
       
       seqIdx = seqIdx + 1;
       
       AllTruePositives    = [AllTruePositives TruePositives];%#ok

   end
end
%%
nbPos = 0;
nbNeg = 0;
seqIdx = 1;

for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        nbPos = nbPos + size(PositivePairsPerSeq{seqIdx}, 1);
        nbNeg = nbNeg + size(NegativePairsPerSeq{seqIdx}, 1);
        seqIdx = seqIdx + 1;
    end
end
disp(Magnification);
disp(['nb of positive/similar pairs of detection is ' int2str(nbPos)])
disp(['nb of negative/dissimilar pairs of detection is ' int2str(nbNeg)])
disp('========================================')
%% compute EMD distances using normalized histograms

% listOfNegToUse = randi(nbNeg, nbPos, 1);
% save(['randSubSelection' Magnification], 'listOfNegToUse');
load(['randSubSelection' Magnification]);


NUMBER_OF_BINS = 32;
punishMatrix = ones(NUMBER_OF_BINS, NUMBER_OF_BINS);
for i = 1:NUMBER_OF_BINS
    for j = 1:NUMBER_OF_BINS
        punishMatrix(i,j) = abs(i-j);
    end
end


seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = zeros(size(PositivePairsPerSeq{seqIdx}, 1), 1);
        tic
        for k =1:size(PositivePairsPerSeq{seqIdx}, 1)
            
            for j = 1:2
                PositivePairsPerSeq{seqIdx}(k, j).NucleusHistRed = histc(PositivePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities, ...
                                                                        linspace(min(PositivePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities),...
                                                                                 max(PositivePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities),...
                                                                                 NUMBER_OF_BINS));
            end
            PositiveEMDs(k) = emd_hat_gd_metric_mex(PositivePairsPerSeq{seqIdx}(k, 1).NucleusHistRed,...
                                                    PositivePairsPerSeq{seqIdx}(k, 2).NucleusHistRed,...
                                                    punishMatrix,-1);
            
        end
        
        PositiveEMDsPerSequence{seqIdx} = PositiveEMDs;%#ok
        
        NegativeEMDs = zeros(size(NegativePairsPerSeq{seqIdx}, 1), 1);
        for k =1:size(NegativePairsPerSeq{seqIdx}, 1)
            for j = 1:2
                NegativePairsPerSeq{seqIdx}(k, j).NucleusHistRed = histc(NegativePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities, ...
                                                                        linspace(min(NegativePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities),...
                                                                                 max(NegativePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities),...
                                                                                 NUMBER_OF_BINS));
            end
            NegativeEMDs(k) = emd_hat_gd_metric_mex(NegativePairsPerSeq{seqIdx}(k, 1).NucleusHistRed,...
                                                    NegativePairsPerSeq{seqIdx}(k, 2).NucleusHistRed,...
                                                    punishMatrix,-1);
        end
        toc
        NegativeEMDsPerSequence{seqIdx} = NegativeEMDs;
        
        seqIdx = seqIdx + 1;
    end
end
%%
PositiveEMDs = [];
NegativeEMDs = [];
seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = [PositiveEMDs; PositiveEMDsPerSequence{seqIdx}];
        NegativeEMDs = [NegativeEMDs; NegativeEMDsPerSequence{seqIdx}];
        seqIdx = seqIdx + 1;
    end
end

NbPoints = 100;
[FP, TP, FN, TN] = getROCRates(PositiveEMDs, NegativeEMDs(listOfNegToUse), NbPoints);
figure; plot(FP, TP);

%% Compute EMD Distances using non normalized global histograms, Red Nuclei
%% Only
NUMBER_OF_BINS = 32;
punishMatrix = ones(NUMBER_OF_BINS, NUMBER_OF_BINS);
for i = 1:NUMBER_OF_BINS
    for j = 1:NUMBER_OF_BINS
        punishMatrix(i,j) = abs(i-j);
    end
end


seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = zeros(size(PositivePairsPerSeq{seqIdx}, 1), 1);
        tic
        for k =1:size(PositivePairsPerSeq{seqIdx}, 1)
            
            for j = 1:2
                PositivePairsPerSeq{seqIdx}(k, j).NucleusHistRed = histc(PositivePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities, ...
                                                                        linspace(CellsPerSeq{seqIdx}(end).MinRed,...
                                                                                 CellsPerSeq{seqIdx}(end).MaxRed,...
                                                                                 NUMBER_OF_BINS));
            end
            PositiveEMDs(k) = emd_hat_gd_metric_mex(PositivePairsPerSeq{seqIdx}(k, 1).NucleusHistRed,...
                                                    PositivePairsPerSeq{seqIdx}(k, 2).NucleusHistRed,...
                                                    punishMatrix,-1);
            
        end
        
        PositiveEMDsPerSequence{seqIdx} = PositiveEMDs;%#ok
        
        NegativeEMDs = zeros(size(NegativePairsPerSeq{seqIdx}, 1), 1);
        for k =1:size(NegativePairsPerSeq{seqIdx}, 1)
            for j = 1:2
                NegativePairsPerSeq{seqIdx}(k, j).NucleusHistRed = histc(NegativePairsPerSeq{seqIdx}(k, j).NucleusRedIntensities, ...
                                                                        linspace(CellsPerSeq{seqIdx}(end).MinRed,...
                                                                                 CellsPerSeq{seqIdx}(end).MaxRed,...
                                                                                 NUMBER_OF_BINS));
            end
            NegativeEMDs(k) = emd_hat_gd_metric_mex(NegativePairsPerSeq{seqIdx}(k, 1).NucleusHistRed,...
                                                    NegativePairsPerSeq{seqIdx}(k, 2).NucleusHistRed,...
                                                    punishMatrix,-1);
        end
        toc
        NegativeEMDsPerSequence{seqIdx} = NegativeEMDs;
        
        seqIdx = seqIdx + 1;
    end
end
%%
PositiveEMDs = [];
NegativeEMDs = [];
seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = [PositiveEMDs; PositiveEMDsPerSequence{seqIdx}];
        NegativeEMDs = [NegativeEMDs; NegativeEMDsPerSequence{seqIdx}];
        seqIdx = seqIdx + 1;
    end
end

NbPoints = 100;
[FP, TP, FN, TN] = getROCRates(PositiveEMDs, NegativeEMDs(listOfNegToUse), NbPoints);
hold on;
plot(FP, TP, '-r');
%% Compute EMD Distances using non normalized global histograms, Green Somata
NUMBER_OF_BINS = 32;
punishMatrix = ones(NUMBER_OF_BINS, NUMBER_OF_BINS);
for i = 1:NUMBER_OF_BINS
    for j = 1:NUMBER_OF_BINS
        punishMatrix(i,j) = abs(i-j);
    end
end


seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = zeros(size(PositivePairsPerSeq{seqIdx}, 1), 1);
        tic
        for k =1:size(PositivePairsPerSeq{seqIdx}, 1)
            
            for j = 1:2
                PositivePairsPerSeq{seqIdx}(k, j).SomaHistGreen = histc(PositivePairsPerSeq{seqIdx}(k, j).SomaGreenIntensities, ...
                                                                        linspace(CellsPerSeq{seqIdx}(end).MinGreen, ...
                                                                                 CellsPerSeq{seqIdx}(end).MaxGreen, ...
                                                                                 NUMBER_OF_BINS));
            end
            PositiveEMDs(k) = emd_hat_gd_metric_mex(PositivePairsPerSeq{seqIdx}(k, 1).SomaHistGreen,...
                                                    PositivePairsPerSeq{seqIdx}(k, 2).SomaHistGreen,...
                                                    punishMatrix,-1);
            
        end
        
        PositiveEMDsPerSequence{seqIdx} = PositiveEMDs;%#ok
        
        NegativeEMDs = zeros(size(NegativePairsPerSeq{seqIdx}, 1), 1);
        for k =1:size(NegativePairsPerSeq{seqIdx}, 1)
            for j = 1:2
                NegativePairsPerSeq{seqIdx}(k, j).SomaHistGreen = histc(NegativePairsPerSeq{seqIdx}(k, j).SomaGreenIntensities, ...
                                                                        linspace(CellsPerSeq{seqIdx}(end).MinGreen,...
                                                                                 CellsPerSeq{seqIdx}(end).MaxGreen,...
                                                                                 NUMBER_OF_BINS));
            end
            NegativeEMDs(k) = emd_hat_gd_metric_mex(NegativePairsPerSeq{seqIdx}(k, 1).SomaHistGreen,...
                                                    NegativePairsPerSeq{seqIdx}(k, 2).SomaHistGreen,...
                                                    punishMatrix,-1);
        end
        toc
        NegativeEMDsPerSequence{seqIdx} = NegativeEMDs;
        
        seqIdx = seqIdx + 1;
    end
end
%%
PositiveEMDs = [];
NegativeEMDs = [];
seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = [PositiveEMDs; PositiveEMDsPerSequence{seqIdx}];
        NegativeEMDs = [NegativeEMDs; NegativeEMDsPerSequence{seqIdx}];
        seqIdx = seqIdx + 1;
    end
end

NbPoints = 100;
[FP, TP, FN, TN] = getROCRates(PositiveEMDs, NegativeEMDs(listOfNegToUse), NbPoints);
hold on;
plot(FP, TP, '-g');