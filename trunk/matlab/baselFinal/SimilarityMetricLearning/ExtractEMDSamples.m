function [PositiveEMDs, NegativeEMDs] = ExtractEMDSamples(Magnification, dataRootDirectory, ConvertedGTRootDir, DetectionDirectory, overlappingTolerance, FastEMD_parameters)

% first collect ground truth data
% which consists in pairs of detection either similar(positive) of
% dissimilar. We assume that preprocessing is done.


listOfGTSeq = dir(dataRootDirectory);
AllTruePositives = [];
disp('========================================')
seqIdx = 1;
nbPos = 0;
nbNeg = 0;


for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        disp('----------------------------------------');
        detectionsFileName = [DetectionDirectory listOfGTSeq(i).name];
        disp(detectionsFileName);
        load(detectionsFileName);
        
        CellsPerSeq{seqIdx} = Cells; %#ok
        CellsListPerSeq{seqIdx} = CellsList; %#ok
        
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
        
        nbPos = nbPos + size(PositivePairsPerSeq{seqIdx}, 1);
        nbNeg = nbNeg + size(NegativePairsPerSeq{seqIdx}, 1);
        
        FalsePositivesPerImage{seqIdx}  = FalsePositives;%#ok
        TruePositivesPerImage{seqIdx}   = TruePositives;%#ok
        
        seqIdx = seqIdx + 1;
        
        AllTruePositives    = [AllTruePositives TruePositives];%#ok
        disp('----------------------------------------');
    end
end

disp(Magnification);
disp(['nb of positive/similar pairs of detections is ' int2str(nbPos)]);
disp(['nb of negative/dissimilar pairs of detections is ' int2str(nbNeg)]);
disp('========================================')

% Now, the EMD distance between histograms of the SomaGreenIntensities is
% computed.

NUMBER_OF_BINS   = FastEMD_parameters.NUMBER_OF_BINS;
THRESH_BINS_DIST = FastEMD_parameters.THRESHOLD_BINS_DIST;
punishMatrix = ones(NUMBER_OF_BINS, NUMBER_OF_BINS);
for i = 1:NUMBER_OF_BINS
    for j = 1:NUMBER_OF_BINS
        punishMatrix(i,j) = abs(i-j);
    end
end

punishMatrix = min(THRESH_BINS_DIST, punishMatrix);

seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = zeros(size(PositivePairsPerSeq{seqIdx}, 1), 1);
        for k =1:size(PositivePairsPerSeq{seqIdx}, 1)
            
            for j = 1:2
                PositivePairsPerSeq{seqIdx}(k, j).SomaHistGreen = histc(PositivePairsPerSeq{seqIdx}(k, j).SomaGreenIntensities, ...
                                                                        linspace(CellsPerSeq{seqIdx}(end).MinGreen, ...
                                                                                 CellsPerSeq{seqIdx}(end).MaxGreen, ...
                                                                                 NUMBER_OF_BINS));%#ok
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
                                                                                 NUMBER_OF_BINS));%#ok
            end
            NegativeEMDs(k) = emd_hat_gd_metric_mex(NegativePairsPerSeq{seqIdx}(k, 1).SomaHistGreen,...
                                                    NegativePairsPerSeq{seqIdx}(k, 2).SomaHistGreen,...
                                                    punishMatrix,-1);
        end
        NegativeEMDsPerSequence{seqIdx} = NegativeEMDs;%#ok
        
        seqIdx = seqIdx + 1;
    end
end

PositiveEMDs = [];
NegativeEMDs = [];
seqIdx = 1;
for i = 1:length(listOfGTSeq)
    if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
        PositiveEMDs = [PositiveEMDs; PositiveEMDsPerSequence{seqIdx}]; %#ok
        NegativeEMDs = [NegativeEMDs; NegativeEMDsPerSequence{seqIdx}]; %#ok
        seqIdx = seqIdx + 1;
    end
end

