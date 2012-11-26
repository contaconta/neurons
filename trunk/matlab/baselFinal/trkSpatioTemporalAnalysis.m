function currentTrack = trkSpatioTemporalAnalysis(currentTrack)

listOfCentroids = [];
listOfTimes     = [];        
for i =1:length(currentTrack)
    listOfCentroids          = vertcat(listOfCentroids,      currentTrack.TimeStep(i).NucleusCentroid);%#ok
    listOfTimes              = [listOfTimes                  currentTrack.TimeStep(i).Time];%#ok
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
    currentTrack.TimeStep(k+1).Acceleration = listOfAccelerations(k);
end

currentTrack.DistanceTraveled = sum(listOfDistances);
currentTrack.MeanSpeed        = mean(listOfSpeeds);
currentTrack.MeanAcceleration = mean(listOfAccelerations);