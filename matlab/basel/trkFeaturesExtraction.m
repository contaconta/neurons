function R = trkFeaturesExtraction(R)

% extracts the features after segmentation/tracking

% time-dependant measurements
disp('...postprocessing - time dependent measures');
R = trkTimeDependentAnalysis(R);

% smooth the statistics
disp('...postprocessing - smoothing the data');
R = trkSmoothAndCleanRun(R);