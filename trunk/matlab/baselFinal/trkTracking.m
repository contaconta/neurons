function Sequence =  trkTracking(folder, resultsFolder, SeqIndexStr, Sample, magnification)
%% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.TIF']);
Rfiles = dir([Rfolder '*.TIF']);

if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps
if TMAX~=length(Gfiles)
   disp(['problem with data in directory: ' folder]);
   return;
end

%% Load the raw data
Red   = trkReadImagesAndNormalize(TMAX, Rfolder);
Green = trkReadImagesAndNormalize(TMAX, Gfolder);
%% preprocess images
disp('...preprocessing');

% frangi parameters
if strcmp(magnification, '10x')
    opt.FrangiScaleRange = [1 2];
elseif strcmp(magnification, '20x')
    opt.FrangiScaleRange = [1 4];
else
    error(['Resolution should be wither 10x or 20x but it is ' magnification]);
end
opt.FrangiScaleRatio = 1;
opt.FrangiBetaOne = .5;
opt.FrangiBetaTwo = 15;
opt.BlackWhite = false;
opt.verbose = false;

tic
Tubularity = trkComputeTubularity(Green, opt);
toc

%% Detect Nuclei
disp('...detecting Nuclei');
% paramaters
% Smoothing the sigma for the red channel
SIGMA_RED         = 2.0;
if strcmp(magnification, '10x')
    MAX_NUCLEUS_AREA  = 170; %  > pi*7*7
    MIN_NUCLEUS_AREA  =  70; %  < pi*5*5
elseif strcmp(magnification, '20x')
    MAX_NUCLEUS_AREA  = 750; %  > pi*15*15
    MIN_NUCLEUS_AREA  = 300; %  < pi*10*10
end
% MaxVariation and Delta are default values from www.vlfeat.org/api/mser_8h.html
MSER_MaxVariation = 0.25;
MSER_Delta        = 2;

tic
Nuclei = trkDetectNuclei(Red, SIGMA_RED, MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA, MSER_MaxVariation, MSER_Delta);
toc


%% detect the Somata using region growing
disp('...detecting somata');

GEODESIC_DISTANCE_THRESH = 2e-6;
if strcmp(magnification, '10x')
    LENGTH_THRESH = 7;
elseif strcmp(magnification, '20x')
    LENGTH_THRESH = 12;
end
STD_MULT_FACTOR = 1.5;

tic
Somata = trkDetectSomataGlobal(Nuclei, Green, GEODESIC_DISTANCE_THRESH, LENGTH_THRESH, STD_MULT_FACTOR);
toc

%% Gather detections into cells
disp('...gather detections into cells');
tic
[Cells CellsList] = trkGatherNucleiAndSomataDetections(Green, Red, Nuclei, Somata);
toc


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
IMAGE_SIZE           = size(Green{1});
DISTANCE_TO_BOUNDARY = 30;
tic
[Cells, tracks, trkSeq, ~] = trkKShortestPaths(CellsList, Cells, ...
                                               TEMPORAL_WIN_SIZE, ...
                                               SPATIAL_WINDOWS_SIZE, ...
                                               MIN_TRACK_LENGTH, ...
                                               NB_BEST_TRACKS, ...
                                               IMAGE_SIZE, ...
                                               DISTANCE_TO_BOUNDARY);
toc

% %% directional non maximum suppression
% disp('... directional non maximum suppression')
% numberOfDirections = 8;
% tic
% DNMS = trkDirectionalNonMaximumSuppression(Green, numberOfDirections);
% toc


%% detect and add filaments to cells
disp('...detect filaments and assign them to each somata');
GEODESIC_DISTANCE_NEURITE_THRESH = 0.6;

tic
[Cells] = trkDetectAndAddFilamentsToCells(Cells, Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH);
toc

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

Cells = rmfield(Cells, 'Neurites');

%% reorganize data
disp('...reorganizing data ')
tic
Sequence = trkReorganizeDataStructure(folder, Rfiles, Gfiles, Green, Red, Sample, SeqIndexStr, Cells, trkSeq);
toc
%%
save([resultsFolder SeqIndexStr],  'Sequence');
