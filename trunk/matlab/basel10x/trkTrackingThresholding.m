function G =  trkTrackingThresholding(folder, resultsFolder, SeqIndexStr, Sample, params)

% get the experiment label, data, and assay position
% [date_txt] = trkGetDateAndLabel(folder);
date_txt = '';
label_txt = Sample;
num_txt = SeqIndexStr;

RECURSIONLIMIT = 5000;
set(0,'RecursionLimit',RECURSIONLIMIT);


%% PARAMETER SETTING (override from command line, read from param file, or default)
paramfile = [resultsFolder SeqIndexStr 'params.mat'];

if nargin > 4
    W_THRESH = params(1);
    %SOMA_PERCENT_THRESH = params(2);
    FRANGI_THRESH = params(2);
    TARGET_NUM_OBJECTS = params(3);
    SOMA_THRESH = params(4);
    disp('  OVERRIDING PARAMETERS!');
else
    if exist(paramfile, 'file');
     	load(paramfile);
        disp(['reading from ' folder 'params.mat']);
    end
end

% default parameters

if ~exist('WT', 'var');                 WT = 50; end;
if ~exist('WSH', 'var');                WSH = 40; end;
if ~exist('W_THRESH', 'var');           W_THRESH = 200; end;
if ~exist('WIN_SIZE', 'var');           WIN_SIZE = 4; end;
if ~exist('FRANGI_THRESH', 'var');      FRANGI_THRESH = .0000001; end; % FRANGI_THRESH = .0000001; end; FRANGI_THRESH = .0000005; end;
if ~exist('NUC_MIN_AREA', 'var');       NUC_MIN_AREA = 5; end; % TODO, it was 150 at the 20x resolution
if ~exist('TARGET_NUM_OBJECTS', 'var'); TARGET_NUM_OBJECTS = 5; end; % TODO, it was 6.5 (= 26/ (2*2)) at the 20x resoltuion
if ~exist('NUC_INT_THRESH', 'var');     NUC_INT_THRESH = .25; end;
if ~exist('SOMA_THRESH', 'var');        SOMA_THRESH = 100; end; %250; end;
if ~exist('MAX_NUCLEUS_AREA', 'var');   MAX_NUCLEUS_AREA = 155; end;%TODO pi*7*7
if ~exist('MIN_NUCLEUS_AREA', 'var');   MIN_NUCLEUS_AREA = 80; end;%TODO  pi*5*5

% other parameters
%TMAX = 20;
MIN_TRACK_LENGTH = 20;               % minimum number of detections for a valid neuron track
SHOW_FALSE_DETECTS = 0;             % show false detections
DISPLAY_FIGURES = 0;                % display figure with results

% image adjustment parameters
% to adjust (normalize) intensities
G_MED = 2537;
G_STD = 28.9134;
G_MAX = 11234;
R_MED = 205;
R_STD = 3.0508;
R_MAX = 327;

% smoothing parameters
sigma_red = 1;
sigma_log_red = 6;

% frangi parameters
opt.FrangiScaleRange = [1 1.5];
opt.FrangiScaleRatio = 1;
opt.FrangiBetaOne = .5;
opt.FrangiBetaTwo = 15;
opt.BlackWhite = false;
opt.verbose = false;

% display important parameters
disp(' -------------------------------------- ');
disp([' W_THRESH             = ' num2str(W_THRESH)]);
disp([' FRANGI_THRESH        = ' num2str(FRANGI_THRESH)]);
disp([' TARGET_NUM_OBJECTS   = ' num2str(TARGET_NUM_OBJECTS)]);
disp([' SOMA_THRESH =        = ' num2str(SOMA_THRESH)]);
disp(' -------------------------------------- ');

% get a list of colors to use for display
cols = color_list();

% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.TIF']);%#ok
Rfiles = dir([Rfolder '*.TIF']);
%experiment1_w2LED red_s1_t26

if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps


%% important data structures
D = [];                     % structure containing nucleus detections
Dlist = cell(1,TMAX);       % cell containing detections indexes in each time step
count = 1;                  % detection counter

%% Load the raw data
R = trkReadImages(TMAX, Rfolder);
G = trkReadImages(TMAX, Gfolder);
%% preprocess images
disp('...preprocessing images');
tic;
LoG = fspecial('log',[5*sigma_log_red 5*sigma_log_red], sigma_log_red);    % Laplacian filter kernel used to find nuclei
[log1, f, J] = preprocessImagesThresholding(R, G, LoG, opt);
dt = toc;
disp(['computation time for preprocessing ' num2str(dt)]);
%%
% estimate the best threshold for detecting nuclei
BEST_LOG_THRESH = getBestLogThresh(log1, NUC_MIN_AREA, TARGET_NUM_OBJECTS);

%% collect nucleus detections
disp('...detecting nuclei');
for  t = 1:TMAX
    M{t} = getNucleiBinaryMask(log1{t}, BEST_LOG_THRESH, NUC_MIN_AREA);
    L = bwlabel(M{t});
    detections_t = regionprops(L, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    
    % add some measurements, create a list of detections
    if ~isempty(detections_t)
        for i = 1:length(detections_t)
            detections_t(i).MeanGreenIntensity = sum(G{t}(detections_t(i).PixelIdxList))/detections_t(i).Area;
            detections_t(i).MeanRedIntensity = sum(R{t}(detections_t(i).PixelIdxList))/detections_t(i).Area;
            
            detections_t(i).Time = t;
            if count == 1
                D = detections_t(i);
            else
                D(count) = detections_t(i);
            end
            Dlist{t} = [Dlist{t} count];
            count = count + 1;
        end
    end
end
toc
%% create the adjacency matrix for all nearby detections
Ndetection = count-1;
A = make_adjacency(Dlist, WIN_SIZE, Ndetection);

% fill out all the distances in the adjacency matrix
edges = find(A == 1);
W = A;
for i = 1:length(edges)
    [r,c] = ind2sub(size(A), edges(i));
    W(r,c) = trkDetectionDistance(D(r), D(c), WT, WSH);
end


%% apply the greedy tracking algorithm to link detections
disp('...greedy tracking');
T = trkGreedyConnect2(W,A,D,W_THRESH);


%% get the track labels from T assigned to each detection
%disp('...graph coloring');
[T tracks] = trkGraphColoring(T, MIN_TRACK_LENGTH); %#ok<*ASGLU>


%% assign ID's to each detections
for t = 1:TMAX
    for d = 1:length(Dlist{t}) % loop through detections in this time step
        detect_ind = Dlist{t}(d);
        D(detect_ind).ID = tracks(detect_ind); %#ok<*AGROW>
    end
end


%% get a list of detections and associated time steps for each track
[trkSeq, timeSeq] = getTrackSequences(Dlist, tracks, D);


%% remove any bad tracks
[D tracks trkSeq timeSeq] = trkRemoveBadTracks(D, tracks, trkSeq, timeSeq, MAX_NUCLEUS_AREA);
tNucleus = toc;
disp(['...elapsed time for nucleus detection and tracking is ' num2str(tNucleus)])

%% render results on the video
disp('...rendering images');
mvNucleus = trkRenderImagesNucleus(TMAX, G, date_txt, num_txt, label_txt, cols,  Dlist, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);
%%
% make a movie of the results
movfile = [SeqIndexStr 'nucTrack'];
trkMovie(mvNucleus, resultsFolder, resultsFolder, movfile); fprintf('\n');

%% generate a list of colors for rendering the results
function cols = color_list()

% cols1 = summer(6);
% cols1 = cols1(randperm(6),:);
% cols2 = summer(8);
% cols2 = cols2(randperm(8),:);
% cols3 = summer(180);
% cols3 = cols3(randperm(180),:);
% cols = [cols1; cols2; cols3];

cols1 = jet(6);
cols1 = cols1(randperm(6),:);
cols2 = jet(8);
cols2 = cols2(randperm(8),:);
cols3 = jet(250);
cols3 = cols3(randperm(250),:);
cols = [cols1; cols2; cols3];