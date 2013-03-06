function Sequence = trkReorganizeDataStructure(folder, Rfiles, Gfiles, Green, Red, Sample, SequenceIdx,...
                                               Cells, trkSeq, ...
                                               TrackedNeurites, TrackedNeuritesList, trkNSeq, timeNSeq)


numberOfTracks = 0;
for i =1:length(trkSeq)
    if ~isempty(trkSeq{i})
        numberOfTracks = numberOfTracks + 1;
    end
end

% start with global informations
Sequence = [];
Sequence.numberOfTracks       = numberOfTracks;
Sequence.InputRootDir         = folder;
Sequence.RedImageFilenames    = Rfiles;
Sequence.GreenImageFilenames  = Gfiles;
Sequence.NumberOfFrames       = length(Rfiles);
Sequence.Sample               = Sample;
Sequence.SeqIdx               = str2double(SequenceIdx);
Sequence.DateProcessed        = date;
Sequence.Plate                = 'TODO';
Sequence.Well                 = 'TODO';
Sequence.Site                 = 'TODO';
Sequence.Entropy              = zeros(1, length(Green));
Sequence.ImgDiffRed           = zeros(1, length(Green));
Sequence.ImgDiffGreen         = zeros(1, length(Green));

Entr        = zeros(1, length(Green));
ImDiffRed   = zeros(1, length(Green));
ImDiffGreen = zeros(1, length(Green));
parfor k = 1:length(Green)
    Entr(k)= entropy(Green{k});
    if k == 1
        ImDiffRed(k)   = 0;
        ImDiffGreen(k) = 0;
    else
        ImDiffRed(k)   = sum(sum(imabsdiff(Red{k}, Red{k-1}))) / numel(Red{k});%#ok
        ImDiffGreen(k) = sum(sum(imabsdiff(Green{k}, Green{k-1}))) / numel(Green{k});%#ok
    end
end

Sequence.Entropy              = Entr;
Sequence.ImgDiffRed           = ImDiffRed;
Sequence.ImgDiffGreen         = ImDiffGreen;

% now the tracks
Sequence.TrackedCells = [];

%% tracked neurites
NeuriteTrackId = 1;
NbTrackedNeurites = 0;
for i = 1:length(trkNSeq)
    if ~isempty(trkNSeq{i})
        NbTrackedNeurites = NbTrackedNeurites + 1;
    end
end
listOfNeuriteTracks = cell(NbTrackedNeurites, 1);
for i = 1:length(trkNSeq)
    if ~isempty(trkNSeq{i})
        ListOfNeuritesInTrack = [];
        for j = 1:length(trkNSeq{i})
            currentNeurite                                           = TrackedNeurites(trkNSeq{i}(j));
            CellIdx                                                  = currentNeurite.CellIdx;
            NeuriteIdx                                               = currentNeurite.NeuriteIdx;
            Cells(CellIdx).NeuritesList(NeuriteIdx).isTracked        = true;
            Cells(CellIdx).NeuritesList(NeuriteIdx).NeuriteTrackId   = NeuriteTrackId;
            ListOfNeuritesInTrack = [ListOfNeuritesInTrack, currentNeurite];%#ok
            if j ==1
                CellTrackIdx                                             = currentNeurite.CellTrackId;
            elseif j > 1 && CellTrackIdx ~= currentNeurite.CellTrackId
                keyboard;
                error('cell trk id problem in neurites track');
            end
        end
        
%         ListOfNeuritesInTrack(1).DelatTotalCableLength     = [];
%         ListOfNeuritesInTrack(1).DeltaLengthBranchesMean   = [];
%         ListOfNeuritesInTrack(1).DeltaMaxExtremeLength     = [];
%         ListOfNeuritesInTrack(1).DeltaExtremeLengthMean    = [];
%         ListOfNeuritesInTrack(1).DeltaMeanLeafLength       = [];
%         ListOfNeuritesInTrack(1).DeltaNbBranches           = [];
%         for j = 2:length(trkNSeq{i})
%             deltaTime  = ListOfNeuritesInTrack(j).Time - ListOfNeuritesInTrack(j-1).Time;
%             ListOfNeuritesInTrack(j).DelatTotalCableLength   = (ListOfNeuritesInTrack(j).TotalCableLength   - ListOfNeuritesInTrack(j-1).TotalCableLength) / deltaTime;%#ok
%             ListOfNeuritesInTrack(j).DeltaLengthBranchesMean = (ListOfNeuritesInTrack(j).LengthBranchesMean - ListOfNeuritesInTrack(j-1).LengthBranchesMean) / deltaTime;%#ok
%             ListOfNeuritesInTrack(j).DeltaMaxExtremeLength   = (ListOfNeuritesInTrack(j).MaxExtremeLength   - ListOfNeuritesInTrack(j-1).MaxExtremeLength) / deltaTime;%#ok
%             ListOfNeuritesInTrack(j).DeltaExtremeLengthMean  = (ListOfNeuritesInTrack(j).ExtremeLengthMean  - ListOfNeuritesInTrack(j-1).ExtremeLengthMean) / deltaTime;%#ok
%             ListOfNeuritesInTrack(j).DeltaMeanLeafLength     = (ListOfNeuritesInTrack(j).MeanLeafLength     - ListOfNeuritesInTrack(j-1).MeanLeafLength) / deltaTime;%#ok
%             ListOfNeuritesInTrack(j).DeltaNbBranches         = (length(ListOfNeuritesInTrack(j).Branches)   - length(ListOfNeuritesInTrack(j-1).Branches)) / deltaTime;%#ok
%         end
        
        NeuritesTrack.Neurites     = ListOfNeuritesInTrack;
        NeuritesTrack.CellTrackIdx = CellTrackIdx;
        listOfNeuriteTracks{NeuriteTrackId} = NeuritesTrack;
        NeuriteTrackId = NeuriteTrackId + 1;
    end
end

%% Tracked cell bodies
for i = 1:length(trkSeq)
    if ~isempty(trkSeq{i})
        currentTrack = [];
        % add some statistics
        currentTrack.LifeTime = length(trkSeq{i});
        
        currentTrack.TimeStep = [];
        
        %% time step level
        for j = 1:length(trkSeq{i})
            currentTrackedCellTimeStep = Cells(trkSeq{i}(j));
            
            % gather the neurites-based mesurments
            LengthBranches               = [];
            LeafLengthBranches           = [];
            ExtremeLength                = [];
            TotalCableLengthPerNeurite   = [];
            ComplexityPerNeurite         = [];
            NbBranchesPerNeurite         = [];
            MaxExtremeLengthPerNeurite   = [];
            for k = 1:length(currentTrackedCellTimeStep.NeuritesList)
                LengthBranches               = [LengthBranches             currentTrackedCellTimeStep.NeuritesList(k).LengthBranches];%#ok
                LeafLengthBranches           = [LeafLengthBranches         currentTrackedCellTimeStep.NeuritesList(k).LeafLengthBranches];%#ok
                ExtremeLength                = [ExtremeLength              currentTrackedCellTimeStep.NeuritesList(k).ExtremeLength];%#ok
                TotalCableLengthPerNeurite   = [TotalCableLengthPerNeurite currentTrackedCellTimeStep.NeuritesList(k).TotalCableLength];%#ok
                ComplexityPerNeurite         = [ComplexityPerNeurite       currentTrackedCellTimeStep.NeuritesList(k).Complexity];%#ok
                NbBranchesPerNeurite         = [NbBranchesPerNeurite       length(currentTrackedCellTimeStep.NeuritesList(k).LengthBranches)];%#ok
                MaxExtremeLengthPerNeurite   = [MaxExtremeLengthPerNeurite max(currentTrackedCellTimeStep.NeuritesList(k).ExtremeLength)];%#ok
            end
            
            currentTrackedCellTimeStep.AllNeurites_LengthBranches          = LengthBranches;
            currentTrackedCellTimeStep.AllNeurites_LeafLengthBranches      = LeafLengthBranches;
            currentTrackedCellTimeStep.AllNeurites_ExtremeLength           = ExtremeLength;
            currentTrackedCellTimeStep.TotalCableLengthsPerNeurite         = TotalCableLengthPerNeurite;
            currentTrackedCellTimeStep.ComplexityPerNeurite                = ComplexityPerNeurite;
            currentTrackedCellTimeStep.NbBranchesPerNeurite                = NbBranchesPerNeurite;
            currentTrackedCellTimeStep.MaxExtremeLengthPerNeurite          = MaxExtremeLengthPerNeurite;
            fieldsToQuantile = {'AllNeurites_LengthBranches', 'AllNeurites_LeafLengthBranches', 'AllNeurites_ExtremeLength', ...
                                'TotalCableLengthsPerNeurite', 'ComplexityPerNeurite', ...
                                'NbBranchesPerNeurite', 'MaxExtremeLengthPerNeurite'};
            quantilesList    = [0 0.25, 0.5, 0.75 1];
            
            currentTrackedCellTimeStep = trkComputeQuantilesAndMean(currentTrackedCellTimeStep, fieldsToQuantile, quantilesList);
            
            currentTrackedCellTimeStep.TotalNeuritesLength   = sum(currentTrackedCellTimeStep.TotalCableLengthsPerNeurite);
            currentTrackedCellTimeStep.TotalNeuritesBranches = sum(currentTrackedCellTimeStep.NbBranchesPerNeurite);
            currentTrackedCellTimeStep.TotalComplexity       = currentTrackedCellTimeStep.TotalNeuritesBranches / ...
                                                               currentTrackedCellTimeStep.TotalNeuritesLength;
            
            currentTrack.TimeStep    = [currentTrack.TimeStep        currentTrackedCellTimeStep];
        end
        
        %% speed, displacement and acceleration
        
        currentTrack = trkSpatioTemporalAnalysis(currentTrack);
        %% temporal analysis for other features (not considering tracked neurites)
        fieldsToAnalyse = { 'NucleusArea',...
                            'NucleusEccentricity', ...
                            'NucleusMajorAxisLength', ...
                            'NucleusMinorAxisLength', ...
                            'NucleusOrientation', ...
                            'NucleusPerimeter', ...
                            'NucleusCircularity', ...
                            'NucleusMeanRedIntensity', ...
                            'NucleusMeanGreenIntensity', ...
                            'SomaArea', ...
                            'SomaEccentricity', ...
                            'SomaMajorAxisLength', ...
                            'SomaMinorAxisLength', ...
                            'SomaOrientation', ...
                            'SomaPerimeter', ...
                            'SomaCircularity', ...
                            'SomaMeanGreenIntensity',...
                            'NumberOfNeurites', ...
                            'TotalNeuritesLength', ...
                            'TotalNeuritesBranches', ...
                            'TotalComplexity', ...
                            'AllNeurites_ExtremeLength_q_0', ...
                            'AllNeurites_ExtremeLength_q_25', ...
                            'AllNeurites_ExtremeLength_q_50', ...
                            'AllNeurites_ExtremeLength_q_75', ...
                            'AllNeurites_ExtremeLength_q_100'};
        currentTrack = trkTemporalAnalysis(currentTrack, fieldsToAnalyse);
        
        
        ListOfNeuriteTracksAssociatedToCellTrack = [];
        for k = 1:NbTrackedNeurites
            if currentTrack.TimeStep(1).ID == listOfNeuriteTracks{k}.CellTrackIdx
                ListOfNeuriteTracksAssociatedToCellTrack = [ListOfNeuriteTracksAssociatedToCellTrack  listOfNeuriteTracks{k}];%#ok
            end
        end
        
        neuriteFieldsToAnalyse = {  'MaxExtremeLength',...
                                    'MeanBranchLength', ...
                                    'MeanLeafLength', ...
                                    'TotalCableLength', ...
                                    'LengthBranches_q_0', ...
                                    'LengthBranches_q_25', ...
                                    'LengthBranches_q_50', ...
                                    'LengthBranches_q_75', ...
                                    'LengthBranches_q_100', ...
                                    'LengthBranchesMean', ...
                                    'ExtremeLength_q_0', ...
                                    'ExtremeLength_q_25', ...
                                    'ExtremeLength_q_50', ...
                                    'ExtremeLength_q_75', ...
                                    'ExtremeLength_q_100', ...
                                    'ExtremeLengthMean', ...
                                    'LeafLengthBranches_q_0',...
                                    'LeafLengthBranches_q_25', ...
                                    'LeafLengthBranches_q_50', ...
                                    'LeafLengthBranches_q_75', ...
                                    'LeafLengthBranches_q_100', ...
                                    'LeafLengthBranchesMean', ...
                                    'MeanGreenIntensities'};
                                
        ListOfNeuriteTracksAssociatedToCellTrack = trkTemporalAnalysisNeurites(ListOfNeuriteTracksAssociatedToCellTrack,...
                                                                               neuriteFieldsToAnalyse);
        
        currentTrack.ListOfNeuriteTracks        = ListOfNeuriteTracksAssociatedToCellTrack;
        currentTrack.NumberOfTrackedNeurites    = length(ListOfNeuriteTracksAssociatedToCellTrack);
        
        Sequence.TrackedCells = [Sequence.TrackedCells currentTrack];
    end
end
