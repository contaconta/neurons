function Green =  trkTrackingMSER(folder, resultsFolder, SeqIndexStr, Sample)

disp(' ======================================== ');

% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.TIF']);
Rfiles = dir([Rfolder '*.TIF']);

if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps
if TMAX~=length(Gfiles)
   disp(['problem with data in directory: ' folder]);
   Green = [];
   return;
end

%% Load the raw data
Red   = trkReadImagesAndNormalize(TMAX, Rfolder);
Green = trkReadImagesAndNormalize(TMAX, Gfolder);
%% preprocess images
disp('...preprocessing images');

% frangi parameters
% opt.FrangiScaleRange = [1 2];
opt.FrangiScaleRange = [1 2];
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
SIGMA_RED         = 2.0;
MAX_NUCLEUS_AREA  = 170; %  > pi*7*7
MIN_NUCLEUS_AREA  =  70; %  < pi*5*5
MSER_MaxVariation = 0.25;
MSER_Delta        = 2;

tic
Nuclei = trkDetectNuclei(Red, SIGMA_RED, MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA, MSER_MaxVariation, MSER_Delta);
toc


%% detect the Somata using region growing
disp('...detecting somata');

GEODESIC_DISTANCE_THRESH = 2e-6;
LENGTH_THRESH = 7;
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
WIN_SIZE = 4;
WT = 50; 
WSH = 40;
W_THRESH = 200;
MIN_TRACK_LENGTH = 20;
SPATIAL_DIST_THRESH = 50;

tic
[Cells, tracks, trkSeq, ~] = trkGenerateNucleiGraphAndTrack(CellsList, Cells, WIN_SIZE, WT, WSH, W_THRESH, MIN_TRACK_LENGTH, SPATIAL_DIST_THRESH);
toc 

%% detect and add filaments to cells
disp('...detect filaments and assign them to each somata');
GEODESIC_DISTANCE_NEURITE_THRESH = 0.6;

tic
[Cells] = trkDetectAndAddFilamentsToCells(Cells, Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH);
toc

%% reorganize data
disp('...reorganizing data ')
tic
Sequence = trkReorganizeDataStructure(Rfiles, Gfiles, Green, Sample, SeqIndexStr, Cells, trkSeq);
toc
%%
save([resultsFolder SeqIndexStr],  'Sequence');
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

%%
% mv = trkRenderImages3(TMAX, G, D, Soma, Dlist, branches);
% % make a movie of the results
% movfile = [SeqIndexStr 'noTRK'];
% trkMovie(mv, resultsFolder, resultsFolder, movfile); fprintf('\n');

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
%keyboard;
% 
% %%
% %keyboard;
% clear g;
% cleaor SL;
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

