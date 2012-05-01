function R = trkPostProcessing(R, Greens)

%TODO:handle case when no filaments are found

% label the filopodia
disp('...postprocessing - labeling filopodia');
for d = 1:length(R.D)
    R.FILAMENTS = trkFindFilopodia(d, R.FILAMENTS, Greens{R.D(d).Time});
end    

% flag the unhappy neurons
disp('...postprocessing - labeling unhappy neurons');
R = HappyNeuronVector(R);

% track the neurites
disp('...postprocessing - tracking neurites');
R = trkTrackNeurites(R);

% time-dependant measurements
disp('...postprocessing - time dependent measures');
R = trkTimeDependentAnalysis(R);

% smooth the statistics
disp('...postprocessing - smoothing the data');
R = trkSmoothAndCleanRun(R);