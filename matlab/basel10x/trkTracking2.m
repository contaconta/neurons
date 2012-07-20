function G =  trkTracking2(folder, resultsFolder, SeqIndexStr, Sample, params)

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
if ~exist('NUC_MIN_AREA', 'var');       NUC_MIN_AREA = 80; end; % TODO, it was 150 at the 20x resolution
if ~exist('TARGET_NUM_OBJECTS', 'var'); TARGET_NUM_OBJECTS = 20; end; % TODO, it was 6.5 (= 26/ (2*2)) at the 20x resoltuion
if ~exist('NUC_INT_THRESH', 'var');     NUC_INT_THRESH = .25; end;
if ~exist('SOMA_THRESH', 'var');        SOMA_THRESH = 100; end; %250; end;
if ~exist('MAX_NUCLEUS_AREA', 'var');   MAX_NUCLEUS_AREA = 650; end;%TODO pi *12 * 12

% other parameters
%TMAX = 20;
MIN_TRACK_LENGTH = 10;               % minimum number of detections for a valid neuron track
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
sigma_log_red = 5;

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
[log1, f, J] = preprocessImages(R, G, LoG, opt);
dt = toc;
disp(['computation time preprocessing is  ' num2str(dt)]);
%%
% estimate the best threshold for detecting nuclei
tic
BEST_LOG_THRESH = getBestLogThresh(log1, NUC_MIN_AREA, TARGET_NUM_OBJECTS);
dt = toc;
disp(['computation time for estimating the threshold is  ' num2str(dt)]);
%% collect nucleus detections
disp('...detecting nuclei');
M = cell(size(R));
parfor  t = 1:TMAX
    M{t} = getNucleiBinaryMask(log1{t}, BEST_LOG_THRESH, NUC_MIN_AREA);
end

for  t = 1:TMAX
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
clear log1;

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
mvNucleus = trkRenderImagesNucleus(TMAX, G, date_txt, num_txt, label_txt, cols, Dlist, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);
%%
% make a movie of the results
movfile = [SeqIndexStr 'nucTrack'];
trkMovie(mvNucleus, resultsFolder, resultsFolder, movfile); fprintf('\n');

% %% detect the Somata using region growing
% disp('...detecting somata');
% %[Soma SMASK SL] = trkDetectSomata(TMAX, Dlist, tracks, D, SOMA_THRESH, J);
% [Soma SL] = trkDetectSomata2(TMAX, Dlist, tracks, D, SOMA_THRESH, J);
% SMASK = zeros(size(SL{1}));
% 
% 
% %%
% mvNucleiSoma = trkRenderImagesNuceliSomata(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, mv, Dlist,  Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS); %#ok<*INUSL>
% % make a movie of the results
% movfile = [SeqIndexStr 'SomaTrack'];
% trkMovie(mvNucleiSoma, resultsFolder, resultsFolder, movfile); fprintf('\n');


% %% find the proper sigmoid parameters to convert Frangi to probabilities
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
% BLANK = zeros(size(mv{1},1), size(mv{1},2));
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
% save(paramfile, 'NUC_INT_THRESH',...
%      'NUC_MIN_AREA',...
%      'WT',...
%      'WSH',...
%      'WIN_SIZE',...
%      'W_THRESH',...
%      'FRANGI_THRESH',...
%      'TARGET_NUM_OBJECTS',...
%      'SOMA_THRESH',...
%      'rmin',...
%      'rmax');
% 
% 
% %% render results on the video
% disp('...rendering images');
% mv = trkRenderImages2(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, mv, Dlist, BLANK, FILAMENTS, Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);
% 
% % make a movie of the results
% movfile = SeqIndexStr;
% trkMovie(mv, resultsFolder, resultsFolder, movfile); fprintf('\n');
% %makemovie(mv, folder, resultsFolder, [  date_txt '_' num_txt '.avi']); disp('');
% 
% 
% %% save everything we need for the analysis
% datafile = [resultsFolder exp_num(i,:) '.mat'];
% trkSaveEssentialData(datafile, D, Dlist, FIL, FILAMENTS, Soma, FrameMeasures, GlobalMeasures, timeSeq, tracks, trkSeq);

% % put everything into a nice structure for the xml writer
% Experiment = makeOutputStructure(D, Soma, Dlist, date_txt, label_txt, tracks, FrameMeasures, num_txt);
% 
% % write the xml file
% xmlFileName = [folder num_txt '.xml'];
% disp(['...writing ' xmlFileName]);
% trkWriteXMLFile(Experiment, xmlFileName);
%matlabpool close;
%keyboard;
% ===================== SUPPORTING FUNCTIONS ==============================
%
%
% =========================================================================


%% detect endpoints and branch points
function [EndP BranchP] = detectEndBranch(F, TMAX, SMASK)

for t = 1:TMAX
    SM = SMASK{t};
    SM = bwmorph(SM, 'dilate');
    EndP{t} = bwmorph(F{t}, 'endpoints');
    BranchP{t} = bwmorph(F{t}, 'branchpoints');
    EndP{t}(SM) = 0;
    BranchP{t}(SM) = 0;
end

%% make time-dependent measurements
function [D Soma] = timeMeasurements(trkSeq, timeSeq, D, Soma)

for i = 1:length(trkSeq)
    dseq = trkSeq{i};
    tseq = timeSeq{i};

    if ~isempty(dseq)

        d1 = dseq(1);
        D(d1).deltaArea = 0;
        D(d1).deltaPerimeter = 0;
        D(d1).deltaMeanGreenIntensity = 0;
        D(d1).deltaEccentricity = 0;
        D(d1).Speed = 0;
        D(d1).Acc = 0;
        D(d1).TravelDistance = 0;

        Soma(d1).deltaArea = 0;
        Soma(d1).deltaPerimeter = 0;
        Soma(d1).deltaMeanGreenIntensity = 0;
        Soma(d1).deltaEccentricity = 0;
        Soma(d1).Speed = 0;
        Soma(d1).Acc = 0;
        Soma(d1).TravelDistance = 0;

        for t = 2:length(dseq)
            d2 = dseq(t);
            d1 = dseq(t-1);
            t2 = tseq(t);
            t1 = tseq(t-1);

            D(d2).deltaArea = D(d2).Area - D(d1).Area;
            D(d2).deltaPerimeter = D(d2).Perimeter - D(d1).Perimeter;
            D(d2).deltaMeanGreenIntensity = D(d2).MeanGreenIntensity - D(d1).MeanGreenIntensity;
            D(d2).deltaEccentricity = D(d2).Eccentricity - D(d1).Eccentricity;
            D(d2).Speed = sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2) / abs(t2 -t1);
            D(d2).Acc = D(d2).Speed - D(d1).Speed;
            D(d2).TravelDistance = D(d1).TravelDistance + sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2 );


            Soma(d2).deltaArea = Soma(d2).Area - Soma(d1).Area;
            Soma(d2).deltaPerimeter = Soma(d2).Perimeter - Soma(d1).Perimeter;
            Soma(d2).deltaMeanGreenIntensity = Soma(d2).MeanGreenIntensity - Soma(d1).MeanGreenIntensity;
            Soma(d2).deltaEccentricity = Soma(d2).Eccentricity - Soma(d1).Eccentricity;
            Soma(d2).Speed = sqrt( (Soma(d2).Centroid(1) - Soma(d1).Centroid(1))^2 + (Soma(d2).Centroid(2) - Soma(d1).Centroid(2))^2) / abs(t2 -t1);
            Soma(d2).Acc = Soma(d2).Speed - Soma(d1).Speed;
            Soma(d2).TravelDistance = Soma(d1).TravelDistance + sqrt( (Soma(d2).Centroid(1) - Soma(d1).Centroid(1))^2 + (Soma(d2).Centroid(2) - Soma(d1).Centroid(2))^2 );

        end
    end
end
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

%% make a movie output of the image sequence
% function makemovie(mv, folder, resultsFolder, outputFilename)
% 
% disp('...writing temporary image files');
% for i = 1:length(mv)
%     imwrite(mv{i}, [folder sprintf('%03d',i) '.png'], 'PNG');
% end
% disp('...encoding movie');
% oldpath = pwd;
% cd(folder);
% ENCODERATE = '5000';
% cmd = ['mencoder "mf://*.png" -mf fps=10 -o ' resultsFolder outputFilename ' -ovc xvid -xvidencopts bitrate=' ENCODERATE ' -really-quiet'];
% % cmd = ['ffmpeg -r 10 -b 600k -i %03d.png ' resultsFolder outputFilename];
% %disp(cmd);
% system(cmd);
% cd(oldpath);
% 
% cmd = ['rm ' folder '*.png'];
% %disp(cmd);
% system(cmd);


%% compute frame-based measurements
function FrameMeasures = getFrameMeasures(G, F, EndP, BranchP, TMAX)

F = F > 0;

for t = 1:TMAX
    Ia = G{t};
    if t == 1
        FrameMeasures.Time = t;
        FrameMeasures.ImgDiff1 = 0;
        FrameMeasures.ImgDiff2 = 0;
        FrameMeasures.Filaments = sum(sum(F{t}));
        FrameMeasures.FilopodiaCount = sum(sum(EndP{t}));
        FrameMeasures.BranchCount = sum(sum(BranchP{t}));
        FrameMeasures.DeltaFilaments = 0;
        FrameMeasures.DeltaFilopodiaCount = 0;
        FrameMeasures.DeltaBranchCount = 0;
    else
        Ib = G{t-1};
        FrameMeasures(t).Time = t;
        FrameMeasures(t).ImgDiff1 = (sum(sum(Ia)) - sum(sum(Ib(:))) ) / sum(sum(Ia));
        FrameMeasures(t).ImgDiff2 =  sum(sum( abs( Ia - Ib))) / sum(sum(Ia));
        FrameMeasures(t).Filaments = sum(sum(F{t}));
        FrameMeasures(t).FilopodiaCount = sum(sum(EndP{t}));
        FrameMeasures(t).BranchCount = sum(sum(BranchP{t}));
        FrameMeasures(t).DeltaFilaments = sum(sum(F{t})) - sum(sum(F{t-1}));
        FrameMeasures(t).DeltaFilopodiaCount = sum(sum(EndP{t})) - sum(sum(EndP{t-1}));
        FrameMeasures(t).DeltaBranchCount = sum(sum(BranchP{t})) - sum(sum(BranchP{t-1}));
    end
    FrameMeasures(t).Entropy = entropy(Ia);
end

% replicate t = 2 for t = 1
FrameMeasures(1) = FrameMeasures(2);


function Experiment = getGlobalMeasures(date_txt,label_txt, tracks, Dlist, num_txt)

Experiment.Date = date_txt;
Experiment.Label = label_txt;
Experiment.NumberOfCells = max(tracks);
Experiment.Length = length(Dlist);
Experiment.AssayPosition = num_txt;


function Experiment = makeOutputStructure(D, Soma, Dlist, date_txt, label_txt, tracks, FrameMeasures, num_txt) %#ok<*DEFNU>

Experiment.Date = date_txt;
Experiment.Label = label_txt;
Experiment.NumberOfCells = max(tracks);
Experiment.Length = length(Dlist);
Experiment.AssayPosition = num_txt;

Soma = rmfield(Soma, 'PixelIdxList');
D = rmfield(D, 'PixelIdxList');
Soma = rmfield(Soma, 'Time');
D = rmfield(D, 'Time');
D = orderfields(D);
Soma = orderfields(Soma);

% loop through the time steps
for t = 1:length(Dlist)

    FM = FrameMeasures(t);
    Experiment.TimeStep(t) = FM;

end

%     Experiment.TimeStep(t).Time             = t;
%     Experiment.TimeStep(t).ImgDiff1         = FrameMeasures(t).ImgDiff1;
%     Experiment.TimeStep(t).ImgDiff2         = FrameMeasures(t).ImgDiff2;
%     Experiment.TimeStep(t).Entropy          = FrameMeasures(t).Entropy;
%     Experiment.TimeStep(t).Filaments        = FrameMeasures(t).Filaments;
%     Experiment.TimeStep(t).FilopodiaCount   = FrameMeasures(t).FilopodiaCount;
%     Experiment.TimeStep(t).BranchCount      = FrameMeasures(t).BranchCount;
%     Experiment.TimeStep(t).DeltaFilaments   = FrameMeasures(t).DeltaFilaments;
%     Experiment.TimeStep(t).DeltaFilopodiaCount = FrameMeasures(t).DeltaFilopodiaCount;
%     Experiment.TimeStep(t).DeltaBranchCount = FrameMeasures(t).DeltaBranchCount;
%    n = 1;

for t = 1:length(Dlist)
    n = 1;
    % loop through all detections in that time step
    for d = 1:length(Dlist{t})
        c = Dlist{t}(d);

        if tracks(c) ~= 0

            if n == 1
                Experiment.TimeStep(t).Neuron.Nucleus = D(c);
                Experiment.TimeStep(t).Neuron.Soma = Soma(c);
            end

            Experiment.TimeStep(t).Neuron(n).Nucleus = D(c);
            Experiment.TimeStep(t).Neuron(n).Soma = Soma(c);
            n = n + 1;

        end
    end
end




