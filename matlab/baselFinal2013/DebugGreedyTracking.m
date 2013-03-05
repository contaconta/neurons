%load DetectionsNew;
%% Generate graph and track
disp('...tracking');
% parameters
TEMPORAL_WIN_SIZE    = 1;
if strcmp(magnification, '10x')
    SPATIAL_WINDOWS_SIZE = 50;
elseif strcmp(magnification, '20x')
    SPATIAL_WINDOWS_SIZE = 100;
end

MIN_TRACK_LENGTH     = 20;
NB_BEST_TRACKS       = 20;

GreedyTrackingParameters.TEMPORAL_WINDOWS_SIZE = TEMPORAL_WIN_SIZE;
GreedyTrackingParameters.SPATIAL_WINDOWS_SIZE  = SPATIAL_WINDOWS_SIZE;
GreedyTrackingParameters.MIN_TRACK_LENGTH      = MIN_TRACK_LENGTH;
GreedyTrackingParameters.NB_BEST_TRACKS        = NB_BEST_TRACKS;
GreedyTrackingParameters.WT                    = 50;
GreedyTrackingParameters.WSH                   = 40;
GreedyTrackingParameters.W_THRESH              = 200;
tic
[Cells, tracks, trkSeq, timeSeq] = trkTrackCellsGreedy(CellsList, Cells, ...
                                                          GreedyTrackingParameters);
toc;


%% render results on the video
disp('...make movie');
% parameters
cols = color_list();

tic
mv = trkRenderImagesAndTracks(Green, Cells, CellsList, tracks, SeqIndexStr, Sample, cols );
% make a movie of the results
movfile = SeqIndexStr ;
trkMovie(mv, resultsFolder, resultsFolder, movfile); fprintf('\n');
toc