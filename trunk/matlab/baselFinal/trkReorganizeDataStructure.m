function Sequence = trkReorganizeDataStructure(folder, Rfiles, Gfiles, Green, Red, Sample, SequenceIdx,...
                                                Cells, trkSeq)


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
        %% temporal analysis for other features
        fieldsToAnalyse = {'NucleusArea',...
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
        
        Sequence.TrackedCells = [Sequence.TrackedCells currentTrack];
    end
end
