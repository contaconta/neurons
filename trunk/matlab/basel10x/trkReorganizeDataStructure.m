function Sequence = trkReorganizeDataStructure(Rfiles, Gfiles, Green, Sample, SequenceIdx,...
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
Sequence.ImgDiff1             = zeros(1, length(Green));
Sequence.ImgDiff2             = zeros(1, length(Green));

for i = 1:length(Green)
    Sequence.Entropy(i) = entropy(Green{i});
end

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
        
        for j = 1:length(trkSeq{i})
            currentTrack.TimeStep    = [currentTrack.TimeStep        Cells(trkSeq{i}(j))];
                
            listGreenSoma            = [listGreenSoma                currentTrack.TimeStep(j).SomaMeanGreenIntensity];%#ok
            
            listRedNucleus           = [listRedNucleus               currentTrack.TimeStep(j).NucleusMeanRedIntensity];%#ok
            
            listOfNucleusArea        = [listOfNucleusArea            currentTrack.TimeStep(j).NucleusArea];%#ok
            listOfSomaArea           = [listOfSomaArea               currentTrack.TimeStep(j).SomaArea];%#ok
            
            listNumberOfNeurites     = [listNumberOfNeurites         currentTrack.TimeStep(j).NeuritesNumber];%#ok
            listOfNeuritesLength     = [listOfNeuritesLength         currentTrack.TimeStep(j).TotalNeuritesLength];%#ok
            listOfNeuritesBranches   = [listOfNeuritesBranches       currentTrack.TimeStep(j).TotalNeuritesBranches];%#ok
            listOfNeuritesComplexity = [listOfNeuritesComplexity     currentTrack.TimeStep(j).MeanNeuritesComplexity];%#ok
            listOfTimes              = [listOfTimes                  currentTrack.TimeStep(j).Time];%#ok
            listOfCentroids          = vertcat(listOfCentroids,      currentTrack.TimeStep(j).NucleusCentroid);%#ok
        end
        
        listOfDisplacements = listOfCentroids(2:end, :) - listOfCentroids(1:end-1, :);
        listOfDistances     = zeros(1, size(listOfDisplacements, 1));
        listOfSpeeds        = zeros(1, size(listOfDisplacements, 1));
        for k=1:length(listOfSpeeds)
            listOfDistances(k) = norm(listOfDisplacements(k, :));
            listOfSpeeds(k) = listOfDistances(k) ./ (listOfTimes(k+1) - listOfTimes(k)); 
            % get distance traveled since last frame
            currentTrack.TimeStep(k+1).DistanceTraveled = listOfDistances(k);
        end
        
        
        currentTrack.MeanGreenIntensity    = mean(listGreenSoma);
        currentTrack.MeanRedNucleus        = mean(listRedNucleus);
        currentTrack.MeanNucleusArea       = mean(listOfNucleusArea);
        currentTrack.MeanSomaArea          = mean(listOfSomaArea);
        currentTrack.MeanNumNeurites       = mean(listNumberOfNeurites);
        currentTrack.DistanceTraveled      = sum(listOfDistances);
        currentTrack.MeanSpeed             = mean(listOfSpeeds);
        currentTrack.MeanNeuritesLength    = mean(listOfNeuritesLength);
        currentTrack.MeanNeuritesBranching = mean(listOfNeuritesBranches);
        currentTrack.MeanNeuritesComplexity= mean(listOfNeuritesComplexity);
        
        Sequence.TrackedCells = [Sequence.TrackedCells currentTrack];
    end
end