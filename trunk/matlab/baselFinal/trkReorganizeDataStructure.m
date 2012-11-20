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
        
%         Sequence.TrackedCells{trkIdx}.TimeStep = [];
        currentTrack.TimeStep = [];
        % intensity based 
        listGreenSoma         = [];
        listRedNucleus        = [];
        % nucleus base
        listOfNucleusArea     = [];
        
        % soma based
        listOfSomaArea        = [];
        
        % neurite based
        listOfNeuritesLength      = [];
        listOfNeuritesBranches   = [];
        listOfNeuritesComplexity = [];
        listNumberOfNeurites     = [];
        
        listOfCentroids          = [];
        listOfTimes              = [];
        
        listMaxExtremeLenght     = [];
        
        for j = 1:length(trkSeq{i})
            currentTrack.TimeStep    = [currentTrack.TimeStep        Cells(trkSeq{i}(j))];
                
            listGreenSoma            = [listGreenSoma                currentTrack.TimeStep(j).SomaMeanGreenIntensity];%#ok
            
            listRedNucleus           = [listRedNucleus               currentTrack.TimeStep(j).NucleusMeanRedIntensity];%#ok
            
            listOfNucleusArea        = [listOfNucleusArea            currentTrack.TimeStep(j).NucleusArea];%#ok
            listOfSomaArea           = [listOfSomaArea               currentTrack.TimeStep(j).SomaArea];%#ok
            
            listNumberOfNeurites     = [listNumberOfNeurites         currentTrack.TimeStep(j).NumberOfNeurites];%#ok
            listOfNeuritesLength     = [listOfNeuritesLength         currentTrack.TimeStep(j).TotalNeuritesLength];%#ok
            listOfNeuritesBranches   = [listOfNeuritesBranches       currentTrack.TimeStep(j).TotalNeuritesBranches];%#ok
            listMaxExtremeLenght     = [listMaxExtremeLenght         currentTrack.TimeStep(j).MaxNeuritesExtremeLength];%#ok
            
            if ~isnan(currentTrack.TimeStep(j).MeanNeuritesComplexity)
                listOfNeuritesComplexity = [listOfNeuritesComplexity     currentTrack.TimeStep(j).MeanNeuritesComplexity];%#ok
            end
            listOfTimes              = [listOfTimes                  currentTrack.TimeStep(j).Time];%#ok
            listOfCentroids          = vertcat(listOfCentroids,      currentTrack.TimeStep(j).NucleusCentroid);%#ok
        end
        
        listOfDisplacements = listOfCentroids(2:end, :) - listOfCentroids(1:end-1, :);
        listOfDistances     = zeros(1, size(listOfDisplacements, 1));
        listOfSpeeds        = zeros(1, size(listOfDisplacements, 1));
        for k=1:length(listOfSpeeds)
            listOfDistances(k) = norm(listOfDisplacements(k, :));
            listOfSpeeds(k) = listOfDistances(k) ./ (listOfTimes(k+1) - listOfTimes(k));
            listOfDisplacements(k, :) = listOfDisplacements(k, :) ./ (listOfTimes(k+1) - listOfTimes(k));
            % get distance traveled since last frame
            currentTrack.TimeStep(k+1).DistanceTraveled = listOfDistances(k);
            % get the speed
            currentTrack.TimeStep(k+1).Speed = listOfSpeeds(k);
        end
        
        listOfAccelerationVectors = listOfDisplacements(2:end, :) - listOfDisplacements(1:end-1, :);
        listOfAccelerations       = zeros(size(listOfAccelerationVectors, 1), 1);
        for k=1:length(listOfAccelerationVectors)
            listOfAccelerations(k) = norm(listOfAccelerationVectors(k, :));
            listOfAccelerations(k) = listOfAccelerations(k) ./ (listOfTimes(k+1) - listOfTimes(k));
            % get the accelerations
            currentTrack.TimeStep(k+1).Acceleration = listOfSpeeds(k);
        end
        
        
        currentTrack.MeanGreenIntensity    = mean(listGreenSoma);
        currentTrack.MeanRedIntensity      = mean(listRedNucleus);
        currentTrack.MeanNucleusArea       = mean(listOfNucleusArea);
        currentTrack.MeanSomaArea          = mean(listOfSomaArea);
        currentTrack.MeanNumNeurites       = mean(listNumberOfNeurites);
        currentTrack.DistanceTraveled      = sum(listOfDistances);
        currentTrack.MeanSpeed             = mean(listOfSpeeds);
        currentTrack.MeanAcceleration      = mean(listOfAccelerations);
        currentTrack.MeanNeuritesLength    = mean(listOfNeuritesLength);
        currentTrack.MeanNeuritesBranching = mean(listOfNeuritesBranches);
        currentTrack.MeanNeuritesComplexity= mean(listOfNeuritesComplexity);
        currentTrack.MeanExtremeLength     = mean(listMaxExtremeLenght);
        
        Sequence.TrackedCells = [Sequence.TrackedCells currentTrack];
    end
end