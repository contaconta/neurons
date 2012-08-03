function G =  trkTrackingMSER(folder, resultsFolder, SeqIndexStr, Sample, params)

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
if ~exist('SPATIAL_DIST_THRESH', 'var');           SPATIAL_DIST_THRESH = 40; end;
if ~exist('FRANGI_THRESH', 'var');      FRANGI_THRESH = .0000001; end; % FRANGI_THRESH = .0000001; end; FRANGI_THRESH = .0000005; end;
if ~exist('SOMA_THRESH', 'var');        SOMA_THRESH = 100; end; %250; end;
if ~exist('MAX_NUCLEUS_AREA', 'var');   MAX_NUCLEUS_AREA = 155; end;%TODO pi*7*7
if ~exist('MIN_NUCLEUS_AREA', 'var');   MIN_NUCLEUS_AREA = 50; end;%TODO  pi*5*5

% other parameters
%TMAX = 20;
MIN_TRACK_LENGTH = 20;               % minimum number of detections for a valid neuron track
SHOW_FALSE_DETECTS = 0;             % show false detections
DISPLAY_FIGURES = 0;                % display figure with results

% smoothing parameters
sigma_red = 2.0;

% frangi parameters
opt.FrangiScaleRange = [1 1.5];
opt.FrangiScaleRatio = 1;
opt.FrangiBetaOne = .5;
opt.FrangiBetaTwo = 15;
opt.BlackWhite = false;
opt.verbose = false;

G_MED = 2537;
G_STD = 28.9134;
G_MAX = 11234;
R_MED = 205;
R_STD = 3.0508;
R_MAX = 327;

% display important parameters
disp(' -------------------------------------- ');
disp([' W_THRESH             = ' num2str(W_THRESH)]);
disp([' FRANGI_THRESH        = ' num2str(FRANGI_THRESH)]);
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
[Rblur,  J, D, Dlist, M, f, branches, count] = preprocessImagesMSER(R, G, sigma_red, MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA, D, Dlist, count);
% [Rblur,  J, D, Dlist, M, f, count] = preprocessImagesMSER(R, G, sigma_red, MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA, D, Dlist, count, opt);
dt = toc;
disp(['computation time for preprocessing and nuclei detection with MSER ' num2str(dt)]);

%% detect the Somata using region growing
disp('...detecting somata');
tic
Soma = trkDetectSomata(D, SOMA_THRESH, J);
toc
% SMASK = zeros(size(SL{1}));
%%
mv = trkRenderImages3(TMAX, G, D, Soma, Dlist, branches);
% make a movie of the results
movfile = [SeqIndexStr 'noTRK'];
trkMovie(mv, resultsFolder, resultsFolder, movfile); fprintf('\n');

%%
% tic
% [D, T, tracks, trkSeq, timeSeq] = trkGenerateNucleiGraphAndTrack(count, Dlist, D, WIN_SIZE, WT, WSH, W_THRESH, MIN_TRACK_LENGTH, SPATIAL_DIST_THRESH);
% 
% 
% %% render results on the video
% % tic
% % disp('...rendering images');
% % mvNucleusSomata = trkRenderImagesNucleiSomata(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, Dlist, Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);
% % 
% % %% make a movie of the results
% % movfile = [SeqIndexStr 'nucTrack'];
% % trkMovie(mvNucleusSomata, resultsFolder, resultsFolder, movfile); fprintf('\n');
% % dt = toc;
% % disp(['computation time for making the video ' num2str(dt)]);
% 
% %% find the proper sigmoid parameters to convert Frangi to probabilities
% t = 50;
% disp('...selecting sigmoid parameters');
% [Asig Bsig] = trkSetSigmoid(SL, f, J, G_STD, G_MAX, Dlist{t});
% clear J;
% 
% 
% %% assign filaments
% %disp('...assigning filament priors');
% priors = assignPriors(D, Dlist, trkSeq, SL, TMAX);
% disp('...assigning filaments'); 
% g = cell(1, TMAX); 
% parfor t = 1:TMAX
%     [FIL{t} g{t}] = assignFilaments(SL{t}, f{t}, Dlist{t}, priors{t}, Asig, Bsig);
%     str = sprintf('   %03d completed', t);
%     disp([str  '     run ' num_txt ' ' date_txt]);
% end
% for t = 1:length(g)
%     f{t} = g{t};
% end
% 
% %%
% %keyboard;
% clear g;
% clear SL;
% 
% %% skeletonize filaments
% disp('...skeletonizing filaments');
% BLANK = zeros(size(R{1},1), size(R{1},2));
% FILAMENTS = trkSkeletonize2(D, FIL, BLANK);
% 
% 
% %% break filaments into neurites
% disp('...breaking skeletons into neurite trees');
% for dd = 1:length(D)
%     ftemp{dd} = f{D(dd).Time};
% end  
% parfor dd = 1:length(D)
%     if D(dd).ID ~= 0
%         set(0,'RecursionLimit',RECURSIONLIMIT);
%         [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites(ftemp{dd}, Soma(dd).PixelIdxList, D(dd).Centroid, FILAMENTS(dd).PixelIdxList);    
%         FILAMENTS(dd).Parents = parents;
%         FILAMENTS(dd).NeuriteID = neuriteId;
%         FILAMENTS(dd).NumKids = branchesLeafs;
%         FILAMENTS(dd).NucleusID = D(dd).ID;
%     end
% end
% 
% %% make time-dependent measurements
% disp('...time-dependent measurements');
% [D Soma] = timeMeasurements(trkSeq, timeSeq, D, Soma);
% %FrameMeasures = getFrameMeasures(G, FIL, EndP, BranchP, TMAX);
% 
% %% get global experiment measures
% GlobalMeasures = getGlobalMeasures(date_txt,label_txt, tracks, Dlist, num_txt);
% 
% toc;
% 
% % save the parameters in the experiment folder
% disp(['...saving parameters to ' paramfile]);
% % save(paramfile, 'NUC_INT_THRESH',...
% %      'NUC_MIN_AREA',...
% %      'WT',...
% %      'WSH',...
% %      'WIN_SIZE',...
% %      'W_THRESH',...
% %      'FRANGI_THRESH',...
% %      'TARGET_NUM_OBJECTS',...
% %      'SOMA_THRESH',...
% %      'rmin',...
% %      'rmax');
% 
% 
% %% render results on the video
% disp('...rendering images');
% mv = trkRenderImages2(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, R, Dlist, BLANK, FILAMENTS, Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);
% 
% % make a movie of the results
% movfile = SeqIndexStr;
% trkMovie(mv, resultsFolder, resultsFolder, movfile); fprintf('\n');
%makemovie(mv, folder, resultsFolder, [  date_txt '_' num_txt '.avi']); disp('');


%% save everything we need for the analysis
% datafile = [resultsFolder exp_num(i,:) '.mat'];
% trkSaveEssentialData(datafile, D, Dlist, FIL, FILAMENTS, Soma, FrameMeasures, GlobalMeasures, timeSeq, tracks, trkSeq);

