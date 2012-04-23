function trkTracking(folder, resultsFolder, SeqIndex, params)

% get the experiment label, data, and assay position
[date_txt, label_txt, num_txt] = trkGetDateAndLabel(folder);

RECURSIONLIMIT = 5000;
set(0,'RecursionLimit',RECURSIONLIMIT);


%% PARAMETER SETTING (override from command line, read from param file, or default)
paramfile = [resultsFolder num2str(SeqIndex) 'params.mat'];

if nargin > 3
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
if ~exist('WT', 'var');                 WT = 50; end;
if ~exist('WSH', 'var');                WSH = 40; end;
if ~exist('W_THRESH', 'var');           W_THRESH = 200; end;
if ~exist('WIN_SIZE', 'var');           WIN_SIZE = 4; end;
if ~exist('FRANGI_THRESH', 'var');      FRANGI_THRESH = .0000001; end; % FRANGI_THRESH = .0000001; end; FRANGI_THRESH = .0000005; end;
if ~exist('NUC_MIN_AREA', 'var');       NUC_MIN_AREA = 150; end;
if ~exist('TARGET_NUM_OBJECTS', 'var'); TARGET_NUM_OBJECTS = 6.5; end;
if ~exist('NUC_INT_THRESH', 'var');     NUC_INT_THRESH = .25; end;
if ~exist('SOMA_THRESH', 'var');        SOMA_THRESH = 100; end; %250; end;
if ~exist('MAX_NUCLEUS_AREA', 'var');   MAX_NUCLEUS_AREA = 2500; end;

% other parameters
%TMAX = 20;
MIN_TRACK_LENGTH = 7;               % minimum number of detections for a valid neuron track
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
rmin = []; rmax = []; % unused variables, kept for stability

% display important parameters
disp(' -------------------------------------- ');
disp([' W_THRESH             = ' num2str(W_THRESH)]);
disp([' FRANGI_THRESH        = ' num2str(FRANGI_THRESH)]);
disp([' TARGET_NUM_OBJECTS   = ' num2str(TARGET_NUM_OBJECTS)]);
disp([' SOMA_THRESH =        = ' num2str(SOMA_THRESH)]);
disp(' -------------------------------------- ');

% frangi parameters
opt.FrangiScaleRange = [1 3];
opt.FrangiScaleRatio = 1;
opt.FrangiBetaOne = .5;
opt.FrangiBetaTwo = 15;
opt.BlackWhite = false;
opt.verbose = false;


% get a list of colors to use for display
cols = color_list();

% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.TIF']);
Rfiles = dir([Rfolder '*.TIF']);
%experiment1_w2LED red_s1_t26

if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps


%% important data structures
D = [];                     % structure containing nucleus detections
Dlist = cell(1,TMAX);       % cell containing detections indexes in each time step
count = 1;                  % detection counter



% get intensity level limits for the sequence
%disp('...reading raw images and determining intensity level limits');
%[rmin rmax gmin gmax R G mv] = readImagesAndGetIntensityLimits(TMAX, Rfolder, Rfiles, Gfolder, Gfiles);
%[rmin rmax gmin gmax R G mvold] = readImagesAndGetIntensityLimits(TMAX, Rfolder, Rfiles, Gfolder, Gfiles);
[R mv] = trkReadAndNormalizeImages(TMAX, Rfolder, 'red', R_MAX, R_STD);
[G mv] = trkReadAndNormalizeImages(TMAX, Gfolder, 'green', G_MAX, G_STD);

%keyboard;

tic;

%% preprocess images
disp('...preprocessing images');
h1 = fspecial('log',[30 30], 6);    % Laplacian filter kernel used to find nuclei
parfor  t1 = 1:TMAX
    %Rt = mat2gray(R{t}, [double(rmin) double(rmax)]);
    Rt = mat2gray(double(R{t1}));
    Rblur{t1} = imgaussian(Rt,2);
    log1{t1} = imfilter(Rt, h1, 'replicate');
    %J{t} = mat2gray(G{t}, [double(gmin) double(gmax)]);
    J{t1} = mat2gray(double(G{t1}));
    f{t1} = FrangiFilter2D(J{t1}, opt);
    %M = J{t} < 3*(G_STD/G_MAX);     % add a mask to get rid of background Frangi resp.
    %f{t} = FrangiFilter2D(J{t}, opt) .* ~M;
end

% estimate the best threshold for detecting nuclei
BEST_LOG_THRESH = getBestLogThresh(log1, NUC_MIN_AREA, TARGET_NUM_OBJECTS);


%keyboard;



%% collect nucleus detections
disp('...detecting nuclei');
for  t = 1:TMAX

    M{t} = getNucleiBinaryMask(log1{t}, Rblur{t}, BEST_LOG_THRESH, NUC_INT_THRESH, NUC_MIN_AREA);
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

    % compute frame-based measurements
    Ia = mv{t}; Ia = Ia(:,:,1);
    if t == 1
        FrameMeasures.ImgDiff1 = 0;
        FrameMeasures.ImgDiff2 = 0;
    else
        Ib = mv{t-1}; Ib = Ib(:,:,1);
        FrameMeasures(t).ImgDiff1 = (sum(sum(Ia)) - sum(sum(Ib(:))) ) / sum(sum(Ia));
        FrameMeasures(t).ImgDiff2 =  sum(sum( abs( Ia - Ib))) / sum(sum(Ia));
    end
    FrameMeasures(t).Entropy = entropy(Ia);
end
clear log1; clear Rblur;



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

%% detect the Somata using region growing
disp('...detecting somata');
%[Soma SMASK SL] = trkDetectSomata(TMAX, Dlist, tracks, D, SOMA_THRESH, J);
[Soma SL] = trkDetectSomata2(TMAX, Dlist, tracks, D, SOMA_THRESH, J);
SMASK = zeros(size(SL{1}));


%% find the proper sigmoid parameters to convert Frangi to probabilities
disp('...selecting sigmoid parameters');
[Asig Bsig] = trkSetSigmoid(SL, f, J, G_STD, G_MAX, Dlist{t});
clear J;


%% assign filaments
%disp('...assigning filament priors');
priors = assignPriors(D, Dlist, trkSeq, SL, TMAX);
disp('...assigning filaments'); 
g = cell(1, TMAX); 
parfor t = 1:TMAX
    [FIL{t} g{t}] = assignFilaments(SL{t}, f{t}, Dlist{t}, priors{t}, Asig, Bsig);
    str = sprintf('   %03d completed', t);
    disp([str  '     run ' num_txt ' ' date_txt]);
end
for t = 1:length(g)
    f{t} = g{t};
end

%keyboard;


clear g;
clear SL;




%% skeletonize filaments
disp('...skeletonizing filaments');
BLANK = zeros(size(mv{1},1), size(mv{1},2));
FILAMENTS = trkSkeletonize2(D, FIL, BLANK);


%% break filaments into neurites
disp('...breaking skeletons into neurite trees');
for dd = 1:length(D)
    ftemp{dd} = f{D(dd).Time};
end  
parfor dd = 1:length(D)
    if D(dd).ID ~= 0
        set(0,'RecursionLimit',RECURSIONLIMIT);
        [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites(ftemp{dd}, Soma(dd).PixelIdxList, D(dd).Centroid, FILAMENTS(dd).PixelIdxList);    
        FILAMENTS(dd).Parents = parents;
        FILAMENTS(dd).NeuriteID = neuriteId;
        FILAMENTS(dd).NumKids = branchesLeafs;
        FILAMENTS(dd).NucleusID = D(dd).ID;
    end
end

%% make time-dependent measurements
disp('...time-dependent measurements');
[D Soma] = timeMeasurements(trkSeq, timeSeq, D, Soma);
%FrameMeasures = getFrameMeasures(G, FIL, EndP, BranchP, TMAX);

%% get global experiment measures
GlobalMeasures = getGlobalMeasures(date_txt,label_txt, tracks, Dlist, num_txt);

toc;

% save the parameters in the experiment folder
disp(['...saving parameters to ' paramfile]);
save(paramfile, 'NUC_INT_THRESH',...
     'NUC_MIN_AREA',...
     'WT',...
     'WSH',...
     'WIN_SIZE',...
     'W_THRESH',...
     'FRANGI_THRESH',...
     'TARGET_NUM_OBJECTS',...
     'SOMA_THRESH',...
     'rmin',...
     'rmax');


%% render results on the video
disp('...rendering images');
mv = trkRenderImages2(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, mv, Dlist, BLANK, FILAMENTS, Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);

% make a movie of the results
movfile = [  date_txt '_' num_txt '.mp4'];
trkMovie(mv, resultsFolder, resultsFolder, movfile); fprintf('\n');
%makemovie(mv, folder, resultsFolder, [  date_txt '_' num_txt '.avi']); disp('');


%% save everything we need for the analysis
datafile = [resultsFolder date_txt '_' num_txt '.mat'];
trkSaveEssentialData(datafile, D, Dlist, FIL, FILAMENTS, Soma, FrameMeasures, GlobalMeasures, timeSeq, tracks, trkSeq);

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



function priors = assignPriors(D, Dlist, trkSeq, SL, TMAX)  %#ok<INUSL>

% priors = cell(size(D));
% for t = 1:TMAX
%     if t == 1
%         priorList = ones(1,numel(Dlist{t})) ./ numel(Dlist{t});
%     else
%         priorList = zeros(1,numel(Dlist{t})) ./ numel(Dlist{t});
%         for i = 1:length(Dlist{t})
%             trkID = D(Dlist{t}(i)).ID;
%             if trkID ~= 0
%                 ind = find(trkSeq{trkID} == Dlist{t}(i));
% 
%                 if ind ~= 1
%                     prevID =  trkSeq{trkID}( ind - 1  );
%                     priorList(i) = sum(sum(SL{t-1} == prevID));
%                 end
%             end
%         end
% 
%         minval = min( priorList( find (priorList))); %#ok<FNDSB>
%         priorList(find(priorList == 0)) = minval; %#ok<FNDSB>
%         % set zeros in the prior list to be the min value
% 
%     end
%     priorList = priorList / sum(priorList);
%     priors{t} = priorList;
% end

priors = cell(size(D));
for t = 1:TMAX
    priorList = ones(1,numel(Dlist{t})) ./ numel(Dlist{t});
    priors{t} = priorList;
end

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


%% estimate the best performing threshold for the LapL nuclei detector
function BEST_LOG_THRESH = getBestLogThresh(log1, NUC_MIN_AREA, TARGET_NUM_OBJECTS)



thresh_list = -.0007:.00005:-.00005;
%Blap = zeros(size(log1{1},1),size(log1{1},2));

for j = 1:length(thresh_list)
    tcount = 1;
    for t = 1:10:length(log1)
        Thresh = thresh_list(j);
        Blap = log1{t} <  Thresh;
        Blap = bwareaopen(Blap, NUC_MIN_AREA);
        Bprop = regionprops(Blap, 'Eccentricity', 'PixelIdxList');
        for i = 1:length(Bprop)
            if Bprop(i).Eccentricity > .85
                Blap(Bprop(i).PixelIdxList) = 0;
            end
        end
        L = bwlabel(Blap);

        det_table(j,tcount) = max(max(L));
        tcount = tcount + 1;
    end

end

dists = abs(mean(det_table,2) - TARGET_NUM_OBJECTS);
[min_val, best_ind] = min(dists);

BEST_LOG_THRESH = thresh_list(best_ind);
disp(['...selected BEST_LOG_THRESH = ' num2str(BEST_LOG_THRESH)]);




%% create and adjacency matrix linking nearby detections
function A = make_adjacency(Dlist, WIN_SIZE, Ndetection)
A = zeros(Ndetection);
for t = 2:length(Dlist)
    for d = 1:length(Dlist{t})
        d_i = Dlist{t}(d);
        min_t = max(1, t-WIN_SIZE);
        for p = min_t:t-1
            A(d_i, Dlist{p}) = 1;
        end
    end
end


%% extract series of track labels and time stamps for each valid track
function [trkSeq, timeSeq] = getTrackSequences(Dlist, tracks, D)
trkSeq = cell(1, max(tracks(:)));
timeSeq = cell(1, max(tracks(:)));
for i = 1:max(tracks(:))

    for t = 1:length(Dlist)
        detections = Dlist{t};
        ids = [D(detections).ID];

        d = detections(find(ids == i,1));

        if ~isempty(d)
            trkSeq{i} = [trkSeq{i} d];
            timeSeq{i} = [timeSeq{i} t];
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
cols3 = jet(180);
cols3 = cols3(randperm(180),:);
cols = [cols1; cols2; cols3];





%% get a binary mask containing nuclei
function B = getNucleiBinaryMask(LAPL, J, LAPL_THRESH, NUC_INT_THRESH, NUC_MIN_AREA)



Blap = LAPL <  LAPL_THRESH;
Blap = bwareaopen(Blap, NUC_MIN_AREA);
Bprop = regionprops(Blap, 'Eccentricity', 'PixelIdxList');
for i = 1:length(Bprop)
    if Bprop(i).Eccentricity > .85
        Blap(Bprop(i).PixelIdxList) = 0;
    end
end
Bint = J > NUC_INT_THRESH;
B = Blap | Bint;
B = bwareaopen(B, NUC_MIN_AREA);    % get rid of small components
B = imfill(B,'holes');              % fill holes




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




