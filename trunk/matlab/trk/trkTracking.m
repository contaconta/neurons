function trkTracking(folder, resultsFolder, params)

% get the experiment label, data, and assay position
[date_txt, label_txt, num_txt] = trkGetDateAndLabel(folder);


%schd = findResource('scheduler', 'configuration', defaultParallelConfig);
%matlabpool(schd)

%% PARAMETER SETTING (override from command line, read from param file, or default)
if nargin > 2
    W_THRESH = params(1);
    %SOMA_PERCENT_THRESH = params(2);
    FRANGI_THRESH = params(2);
    TARGET_NUM_OBJECTS = params(3);
    SOMA_THRESH = params(4);
    disp('  OVERRIDING PARAMETERS!');
else
    if exist([folder 'params.mat' ], 'file');
     	load([folder 'params.mat']);
        disp(['reading from ' folder 'params.mat']);
    end
end
if ~exist('WT', 'var');                 WT = 50; end;
if ~exist('WSH', 'var');                WSH = 40; end;
if ~exist('W_THRESH', 'var');           W_THRESH = 200; end;
if ~exist('WIN_SIZE', 'var');           WIN_SIZE = 4; end;
if ~exist('FRANGI_THRESH', 'var');      FRANGI_THRESH = .0000005; end;
if ~exist('NUC_MIN_AREA', 'var');       NUC_MIN_AREA = 150; end;
if ~exist('TARGET_NUM_OBJECTS', 'var'); TARGET_NUM_OBJECTS = 6.5; end;
if ~exist('NUC_INT_THRESH', 'var');     NUC_INT_THRESH = .25; end;
if ~exist('SOMA_THRESH', 'var');        SOMA_THRESH = 130; end;

% other parameters
MIN_FILAMENT_SIZE = 30;             % minimum size of a neurite/filopod
MIN_TRACK_LENGTH = 7;               % minimum number of detections for a valid neuron track
SHOW_FALSE_DETECTS = 0;             % show false detections
DISPLAY_FIGURES = 1;                % display figure with results


% display important parameters
disp(' -------------------------------------- ');
disp([' W_THRESH             = ' num2str(W_THRESH)]);
disp([' FRANGI_THRESH        = ' num2str(FRANGI_THRESH)]);
disp([' TARGET_NUM_OBJECTS   = ' num2str(TARGET_NUM_OBJECTS)]);
disp([' SOMA_THRESH =        = ' num2str(SOMA_THRESH)]);
disp(' -------------------------------------- ');

% frangi parameters
opt.FrangiScaleRange = [1 2];
opt.FrangiScaleRatio = 1;
opt.FrangiBetaOne = .5;
opt.FrangiBetaTwo = 15;
opt.BlackWhite = false;
opt.verbose = false;



%% add necessary paths
if ~isempty( strfind(path, [pwd '/frangi_filter_version2a']) )
    addpath([pwd '/frangi_filter_version2a/']);
end


% get a list of colors to use for display
cols = color_list();

% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.tif']);
Rfiles = dir([Rfolder '*.tif']);
if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps


%% important data structures
D = [];                     % structure containing nucleus detections
Dlist = cell(1,TMAX);       % cell containing detections indexes in each time step
count = 1;                  % detection counter
FILAMENTS = [];


% get intensity level limits for the sequence
disp('...reading raw images and determining intensity level limits');
[rmin rmax gmin gmax R G mv] = readImagesAndGetIntensityLimits(TMAX, Rfolder, Rfiles, Gfolder, Gfiles);



%% preprocess images
disp('...preprocessing images');
h1 = fspecial('log',[30 30], 6);    % Laplacian filter kernel used to find nuclei
parfor t = 1:TMAX
    Rt = mat2gray(R{t}, [double(rmin) double(rmax)]);
    Rblur{t} = imgaussian(Rt,2);
    log1{t} = imfilter(Rt, h1, 'replicate');
    J{t} = mat2gray(G{t}, [double(gmin) double(gmax)]);
    f{t} = FrangiFilter2D(J{t}, opt);
end




% estimate the best threshold for detecting nuclei
BEST_LOG_THRESH = getBestLogThresh(log1, NUC_MIN_AREA, TARGET_NUM_OBJECTS);


%% collect nucleus detections
disp('...detecting nuclei');
for t = 1:TMAX

    M{t} = getNucleiBinaryMask(log1{t}, Rblur{t}, BEST_LOG_THRESH, NUC_INT_THRESH, NUC_MIN_AREA);
    clear Rblur;
    L = bwlabel(M{t});
    detections_t = regionprops(L, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>

    % add some measurements, create a list of detections
    if ~isempty(detections_t)
        for i = 1:length(detections_t)
            detections_t(i).MeanIntensity = sum(G{t}(detections_t(i).PixelIdxList))/detections_t(i).Area;
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
T = trkGreedyConnect(W,A,D,W_THRESH);


%% get the track labels from T assigned to each detection
disp('...graph coloring');
[T tracks] = trkGraphColoring(T, MIN_TRACK_LENGTH); %#ok<*ASGLU>


%% assign ID's to each detections
for t = 1:TMAX
    % loop through detections in this time step
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);
        D(detect_ind).ID = tracks(detect_ind); %#ok<*AGROW>
    end
end


%% get a list of detections and associated time steps for each track
[trkSeq, timeSeq] = getTrackSequences(Dlist, tracks, D);


%% detect the Somata using region growing
disp('...detecting somata');
[Soma SMASK SL] = detectSomata(TMAX, Dlist, tracks, D, SOMA_THRESH, J);
clear J;


%% assign filaments
disp('...assigning filament priors');
BLANK = zeros(size(G{1}));
priors = cell(size(D));
for t = 1:TMAX
    if t == 1
        priorList = ones(1,numel(Dlist{t})) ./ numel(Dlist{t});
    else
        priorList = zeros(1,numel(Dlist{t})) ./ numel(Dlist{t});
        for i = 1:length(Dlist{t})
            trkID = D(Dlist{t}(i)).ID;
            if trkID ~= 0
                ind = find(trkSeq{trkID} == Dlist{t}(i));

                if ind ~= 1
                    prevID =  trkSeq{trkID}( ind - 1  );
                    priorList(i) = sum(sum(SL{t-1} == prevID));
                end
            end
        end

        minval = min( priorList( find (priorList)));
        priorList(find(priorList == 0)) = minval;
        % set zeros in the prior list to be the min value

    end
    priorList = priorList / sum(priorList);
    priors{t} = priorList;
end

disp('...assigning filaments');
parfor t = 1:TMAX
    FIL{t} = assignFilaments(SL{t}, f{t}, Dlist{t}, priors{t});
    disp(['...' num2str(t) ' completed']); 
end
clear SL f;

disp('...skeletonizing filaments');
FILAMENTS = struct('PixelIdxList',[],'Endpoints',[], 'Branchpoints',[]);
FILAMENTS(length(D)).PixelIdxList = [];
parfor d = 1:length(D)
    if D(d).ID ~= 0
        t = D(d).Time;
        FILi = FIL{t} == d; %#ok<PFBNS>
        FILi = bwmorph(FILi, 'erode', 1);
        FILSKEL = bwmorph(FILi, 'skel', Inf);
        FILAMENTS(d).PixelIdxList = find(FILSKEL);
        FILAMENTS(d).Endpoints = find(bwmorph(FILSKEL, 'endpoints'));
        %FILAMENTS(d).Branchpoints = find(bwmorph(FILSKEL, 'branchpoints'));
        FILAMENTS(d).Branchpoints = find( bwmorph(bwmorph(FILSKEL, 'branchpoints'), 'thin', Inf);
    end
end



%% render results on the video
% 1. draw results on the videos.
% 2. draw text annotations on the image
disp('...rendering images');
mv = trkRenderImages(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, mv, Dlist, BLANK, FILAMENTS, Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS);




%% make time-dependent measurements
disp('...time-dependent measurements');
[D Soma] = timeMeasurements(trkSeq, timeSeq, D, Soma);
FrameMeasures = getFrameMeasures(G, FIL, EndP, BranchP, TMAX);

% put everything into a nice structure for the xml writer
Experiment = makeOutputStructure(D, Soma, Dlist, date_txt, label_txt, tracks, FrameMeasures, num_txt);


% write the xml file
%xmlFileName = [folder num_txt '.xml'];
%disp(['...writing ' xmlFileName]);
%trkWriteXMLFile(Experiment, xmlFileName);


% save the parameters in the experiment folder
paramfile = [resultsFolder date_txt '_' num_txt '.mat'];
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

% make a movie of the results
movfile = [  date_txt '_' num_txt '.avi'];
trkMovie(mv, folder, resultsFolder, movfile); disp('');
%makemovie(mv, folder, resultsFolder, [  date_txt '_' num_txt '.avi']); disp('');


%matlabpool close;


keyboard;



















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




%% detect somata
function [Soma SMASK SL] = detectSomata(TMAX, Dlist, tracks, D, SOMA_THRESH, J)

BLANK = zeros(size(J{1}));
Soma = [];
SL = [];

for t = 1:TMAX
    SMASK{t} = BLANK;
    SL{t} = BLANK;
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        r = max(1,round(D(detect_ind).Centroid(2)));
        c = max(1,round(D(detect_ind).Centroid(1)));
        DET = BLANK;
        DET(D(detect_ind).PixelIdxList) = 1;
        DET = DET > 0;
        SOMA_INT_DIST =  SOMA_THRESH * mean(J{t}(DET));


        % segment the Soma using region growing
        SomaM    = trkRegionGrow3(J{t},DET,SOMA_INT_DIST,r,c);

        % fill holes in the somas, and find soma perimeter
        SomaM  	= imfill(SomaM, 'holes');
        SomaM   = bwmorph(SomaM, 'dilate', 2);

        if tracks(detect_ind) ~= 0
            % collect information about the soma region
            soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
            soma_prop(1).ID = tracks(detect_ind);
            soma_prop(1).Time = t;
            soma_prop(1).MeanIntensity = sum(J{t}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;

            % fill the soma structure
            if isempty(Soma)
                Soma = soma_prop(1);
            end

            % store properties into the Soma struct
            Soma(detect_ind) = soma_prop(1);

            % add the soma to a label mask
            SL{t}(soma_prop(1).PixelIdxList) = detect_ind;
        end

        SMASK{t}(SomaM) = 1;


    end

    SMASK{t} = SMASK{t} > 0;
    %     SMASK{t} = bwmorph(SMASK{t}, 'dilate', 2);
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
        D(d1).deltaMeanIntensity = 0;
        D(d1).deltaEccentricity = 0;
        D(d1).Speed = 0;
        D(d1).Acc = 0;
        D(d1).TravelDistance = 0;

        Soma(d1).deltaArea = 0;
        Soma(d1).deltaPerimeter = 0;
        Soma(d1).deltaMeanIntensity = 0;
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
            D(d2).deltaMeanIntensity = D(d2).MeanIntensity - D(d1).MeanIntensity;
            D(d2).deltaEccentricity = D(d2).Eccentricity - D(d1).Eccentricity;
            D(d2).Speed = sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2) / abs(t2 -t1);
            D(d2).Acc = D(d2).Speed - D(d1).Speed;
            D(d2).TravelDistance = D(d1).TravelDistance + sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2 );


            Soma(d2).deltaArea = Soma(d2).Area - Soma(d1).Area;
            Soma(d2).deltaPerimeter = Soma(d2).Perimeter - Soma(d1).Perimeter;
            Soma(d2).deltaMeanIntensity = Soma(d2).MeanIntensity - Soma(d1).MeanIntensity;
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


%% find the intensity limits of the image sequences, [rmin rmax] [gmin gmax]
function [rmin rmax gmin gmax R G mv] = readImagesAndGetIntensityLimits(TMAX, Rfolder, Rfiles, Gfolder, Gfiles)

rmax = 0;  rmin = 255;  gmax = 0;  gmin = 2^16;
R = cell(1,TMAX);
G = R;
mv = R;


for t = 1:TMAX
    if mod(t,10) == 0
        disp(['   t = ' num2str(t) '/' num2str(TMAX)]);
    end

    R{t} = imread([Rfolder Rfiles(t).name]);
    rmax = max(rmax, max(R{t}(:)));
    rmin = min(rmin, min(R{t}(:)));
    G{t} = imread([Gfolder Gfiles(t).name]);
    gmax = max(gmax, max(G{t}(:)));
    gmin = min(gmin, min(G{t}(:)));

    if t == 1
        lims = stretchlim(G{t});
    end
    G8bits = trkTo8Bits(G{t}, lims);

    % make an output image
    Ir = mat2gray(G8bits);
    I(:,:,1) = Ir;
    I(:,:,2) = Ir;
    I(:,:,3) = Ir;

    mv{t} = I;
end
disp('');




%% generate a list of colors for rendering the results
function cols = color_list()

cols1 = summer(6);
cols1 = cols1(randperm(6),:);
cols2 = summer(8);
cols2 = cols2(randperm(8),:);
cols3 = summer(180);
cols3 = cols3(randperm(180),:);
cols = [cols1; cols2; cols3];


%% convert 16-bit image to 8-bit image
function J = trkTo8Bits(I, lims)

J = imadjust(I, lims, []);
J = uint8(J/2^8);


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


function Experiment = makeOutputStructure(D, Soma, Dlist, date_txt, label_txt, tracks, FrameMeasures, num_txt)

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




