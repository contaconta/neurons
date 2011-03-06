function R = trkPostProcessing(R)



% label the filopodia
disp('...postprocessing - labeling filopodia');
for d = 1:length(R.D)
    R.FILAMENTS = trkFindFilopodia(d, R.FILAMENTS);
end    

% flag the unhappy neurons
disp('...postprocessing - labeling unhappy neurons');
R = HappyNeuronVector(R);

% track the neurites
disp('...postprocessing - tracking neurites');
R = trkTrackNeurites(R);

% time-dependant measurements


% smooth the statistics
disp('...postprocessing - smoothing the data');
R = trkSmoothAndCleanRun(R);