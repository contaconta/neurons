function R = trkTrackingPostProcessing(R)

% this is a copy of trkPostProcessing, just to separate
% segmentation/tracking from features extraction.

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