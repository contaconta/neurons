function trkTrack(folder, params)


% get the experiment label, data, ans assay position
[date_txt, label_txt, num_txt] = trkGetDateAndLabel(folder);

% add the path to the frangi filter
addpath([pwd '/frangi_filter_version2a/']);

% PARAMETER SETTING (override from command line, read from param file, or default)
if nargin > 1
    NUC_INT_THRESH = params(1);
    MIN_SIZE = params(2);
    WT = params(3);            
    WSH = params(4);           
    WIN_SIZE = params(5);      
    W_THRESH = params(6);
    SOMA_PERCENT_THRESH = params(7);
    FRANGI_THRESH = params(8);
    disp('  OVERRIDING PARAMETERS!');
else
    if exist([folder 'params.mat' ], 'file');
     	load([folder 'params.mat']);
        disp(['reading from ' folder 'params.mat']);
        if ~exist('WT', 'var'); WT = 50; end;
        if ~exist('WSH', 'var'); WSH = 40; end;
        if ~exist('W_THRESH', 'var'); W_THRESH = 240; end;
        if ~exist('WIN_SIZE', 'var'); WIN_SIZE = 4; end;
        if ~exist('FRANGI_THRESH', 'var'); FRANGI_THRESH = .0000005; end;
    else        
        NUC_INT_THRESH = .22;       % nucleus intensity thresh (lower finds more nuclei)
        MIN_SIZE = 50;              % minimum nucleus size 
        WT = 50;                    % time-weight for detection distance
        WSH = 40;                   % shape-weight for detection distance
        WIN_SIZE = 4;               % time-search window for tracking
        W_THRESH = 200;             % tracking distance thresh (higher = more difficult tracks are allowed)
        SOMA_PERCENT_THRESH = 0.15; % higher = more variation in soma intensity, larger somas
        FRANGI_THRESH = .0000005;   %
    end
end


MIN_FILAMENT_SIZE = 30;

% get a list of colors to use for display
cols = color_list();

disp( ' ');
disp(' -------------------------------------- ');
disp([' NUC_INT_THRESH       = ' num2str(NUC_INT_THRESH) ]);
disp([' MIN_SIZE             = ' num2str(MIN_SIZE) ]);
disp([' WT                   = ' num2str(WT) ]);
disp([' WSH                  = ' num2str(WSH) ]);
disp([' WIN_SIZE             = ' num2str(WIN_SIZE)]);
disp([' W_THRESH             = ' num2str(W_THRESH)]);
disp([' SOMA_PERCENT_THRESH  = ' num2str(SOMA_PERCENT_THRESH)]);
disp([' FRANGI_THRESH        = ' num2str(FRANGI_THRESH)]);
disp(' -------------------------------------- ');



addpath('/home/ksmith/code/neurons/matlab/geneva/bm3d/');
addpath('/home/ksmith/code/neurons/matlab/toolboxes/kevin/');

Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];

Gfiles = dir([Gfolder '*.tif']);
Rfiles = dir([Rfolder '*.tif']);

TMAX = 5; %length(Rfiles);

%% find the intensity limits of the image sequences, [rmin rmax] [gmin gmax]
if ~exist('rmax', 'var')
    rmax = 0;  rmin = 255;  gmax = 0;  gmin = 2^16;
    for t = 1:TMAX
        R = imread([Rfolder Rfiles(t).name]);
        rmax = max(rmax, max(R(:)));
        rmin = min(rmin, min(R(:)));
        G = imread([Gfolder Gfiles(t).name]);
        gmax = max(gmax, max(G(:)));
        gmin = min(gmin, min(G(:)));
    end
    %rmax = 886;
    %rmin = 199;
end
    

mv = cell(0);
D = []; count = 1; Dlist = cell(1,TMAX);
    


h1 = fspecial('log',[30 30], 6);

%% loop through sequence, collect nucleus detections
disp('...preprocessing images');
for t = 1:TMAX
    
    G = imread([Gfolder Gfiles(t).name]);
    R = imread([Rfolder Rfiles(t).name]);

    if t == 1
        lims = stretchlim(G);
    end
    Gorig = G;
    G = trkTo8Bits(G, lims);
    
    %[NA, RD] = BM3D(1, R, 10);
    RD{t} = mat2gray(R, [double(rmin) double(rmax)]);
    RD{t}=imgaussian(RD{t},2);
    

    %% segment the nuclei from the Red channel
    %M = segment_nuclei(RD{t}, NUC_INT_THRESH, MIN_SIZE);
    log1 = imfilter(RD{t}, h1, 'replicate');
    M = segment_nuclei2(log1, RD{t}, -.0002, NUC_INT_THRESH, MIN_SIZE);
    
    % make an output image
    Ir = mat2gray(G);
    Ig = mat2gray(G);
    Ib = mat2gray(G);     
    I(:,:,1) = Ir;
    I(:,:,2) = Ig;
    I(:,:,3) = Ib;

    % store the output image to a movie
    mv{t} = I;

    % detect nuclei for this time step
    L = bwlabel(M);
    detections_t = regionprops(L, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
       
    % add some measurements, create a list of detections
    if ~isempty(detections_t)
        for i = 1:length(detections_t)
            detections_t(i).MeanIntensity = sum(Gorig(detections_t(i).PixelIdxList))/detections_t(i).Area;
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
    
    
    %% compute frame-based measurements
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

BLANK = zeros(size(I,1), size(I,2));
BLANK = BLANK > 1;



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
[T tracks] = trkGraphColoring(T); %#ok<*ASGLU>

Soma = [];





%% assign ID's to each detections 
for t = 1:TMAX
    % loop through detections in this time step
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);
        D(detect_ind).ID = tracks(detect_ind); %#ok<*AGROW>
    end
end


F = cell(0);

opt.FrangiScaleRange = [1 2];
opt.FrangiScaleRatio = 2;
opt.FrangiBetaOne = .5;
opt.FrangiBetaTwo = 15;
opt.BlackWhite = false;
opt.verbose = false;

h2 = fspecial('log',[30 30], 6);

%% segment the soma and filaments
for t = 1:TMAX
        
    disp(['   t = ' num2str(t)]);
    I = mv{t};
    
    
    %% compute the frangi response
    J = mat2gray(I(:,:,1));
    Jlog = sqrt(J);
    [F{t},S,O] = FrangiFilter2D(Jlog, opt);
    F{t} = F{t} > FRANGI_THRESH;
    F{t} = bwareaopen(F{t}, MIN_FILAMENT_SIZE, 8);
    
%     % remove blobs from the mask
%     B = imfilter(RD{t}, h2, 'replicate');
%     B = B <  -.00009;
%     B = bwareaopen(B, 150);
%     Bprop = regionprops(B, 'Eccentricity', 'PixelIdxList');
%     for i = 1:length(Bprop)
%         if Bprop(i).Eccentricity > .9
%             B(Bprop(i).PixelIdxList) = 0;
%         end
%     end
%     B(1:end,1) = 1; B(1:end,end) = 1; B(1,1:end) = 1; B(end,1:end) = 1;
%     
%    	%B = bwmorph(B, 'thicken', 9);    
%     F{t}(B) = 0;
    
    %keyboard;
    
    % loop through detections in this time step
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);
        
        if tracks(detect_ind) ~= 0
            
            %% 1. perform region growing for the Soma
            r = max(1,round(D(detect_ind).Centroid(2)));
            c = max(1,round(D(detect_ind).Centroid(1)));
            if I(r,c,1) == 1
                %SOMA_INT_DIST = .28;
                %SOMA_INT_DIST = .20;
                %SOMA_INT_DIST = .16;
                SOMA_INT_DIST = SOMA_PERCENT_THRESH * I(r,c,1);
            else
                SOMA_INT_DIST = SOMA_PERCENT_THRESH * I(r,c,1);
            end
            
            % grow the soma
            %J2 = imgaussian(J,2);
            %SomaM    = trkRegionGrow(J2,r,c,SOMA_INT_DIST);
            DET = BLANK; DET(D(detect_ind).PixelIdxList) = 1;
            SomaM    = trkRegionGrow2(J,DET,SOMA_INT_DIST);
            %SomaM = DET;
            
            % fill holes in the somas, and find soma perimeter
            SomaM    = imfill(SomaM, 'holes');
                        
            % collect information about the soma
            soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
            soma_prop(1).ID = tracks(detect_ind);
            soma_prop(1).Time = t;
            soma_prop(1).MeanIntensity = sum(Ir(soma_prop(1).PixelIdxList))/soma_prop(1).Area;
            
            % fill the soma structure
            if isempty(Soma)
                Soma = soma_prop(1);
            end
            
            % remove soma pixels from the filaments
            %F{t}( bwmorph(SomaM, 'thicken', 4)) = 0;
            %F{t}( bwmorph(SomaM, 'thicken', 3)) = 0;
            
            Soma(detect_ind) = soma_prop(1);
        end
    end
    
    %keyboard;
end


%% get a list of detections and associated time steps for each track
[trkSeq, timeSeq] = getTrackSequences(Dlist, tracks, D);

%% make time-dependent measurements
disp('making time-dependent measurements');
for i = 1:length(trkSeq)
    dseq = trkSeq{i};
    tseq = timeSeq{i};
        
    d1 = dseq(1);
    D(d1).deltaArea = 0;
    D(d1).deltaPerimeter = 0;
    D(d1).deltaMeanIntensity = 0;
    D(d1).deltaEccentricity = 0;
    D(d1).Speed = 0;
    D(d1).TravelDistance = 0;
    
    Soma(d1).deltaArea = 0;
    Soma(d1).deltaPerimeter = 0;
    Soma(d1).deltaMeanIntensity = 0;
    Soma(d1).deltaEccentricity = 0;
    Soma(d1).Speed = 0;
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
        D(d2).TravelDistance = D(d1).TravelDistance + sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2 );
        
        
        Soma(d2).deltaArea = Soma(d2).Area - Soma(d1).Area;
        Soma(d2).deltaPerimeter = Soma(d2).Perimeter - Soma(d1).Perimeter;
        Soma(d2).deltaMeanIntensity = Soma(d2).MeanIntensity - Soma(d1).MeanIntensity;
        Soma(d2).deltaEccentricity = Soma(d2).Eccentricity - Soma(d1).Eccentricity;
        Soma(d2).Speed = sqrt( (Soma(d2).Centroid(1) - Soma(d1).Centroid(1))^2 + (Soma(d2).Centroid(2) - Soma(d1).Centroid(2))^2) / abs(t2 -t1);
        Soma(d2).TravelDistance = Soma(d1).TravelDistance + sqrt( (Soma(d2).Centroid(1) - Soma(d1).Centroid(1))^2 + (Soma(d2).Centroid(2) - Soma(d1).Centroid(2))^2 );
        
    end
end



%% render results on the video
% 1. draw results on the videos. 
% 2. draw text annotations on the image
disp('rendering images...');
B = zeros(size(I,1), size(I,2));
for t = 1:TMAX
    
    I = mv{t};
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);
    
    %% 1. draw the objects
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);
        
        if tracks(detect_ind) ~= 0
            % color the nucleus
            Ir(D(detect_ind).PixelIdxList) = cols(tracks(detect_ind),1);
            Ig(D(detect_ind).PixelIdxList) = cols(tracks(detect_ind),2);
            Ib(D(detect_ind).PixelIdxList) = cols(tracks(detect_ind),3);
            
            % color the perimeter of the soma
            SomaM = B;
            SomaM(Soma(detect_ind).PixelIdxList) = 1;
            SomaP = bwperim(SomaM); 
            Ir(SomaP) = max(0, cols(tracks(detect_ind),1) - .2);
            Ig(SomaP) = max(cols(tracks(detect_ind),2) - .2);
            Ib(SomaP) = max(cols(tracks(detect_ind),3) - .2);
        else
            % color the nucleus (false detections!)
            Ir(D(detect_ind).PixelIdxList) = 1;
            Ig(D(detect_ind).PixelIdxList) = 0;
            Ib(D(detect_ind).PixelIdxList) = 0;
        end    
    end
    
    % draw the filaments
    Ir(F{t}) = .8;
    Ig(F{t}) = 0;
    Ib(F{t}) = 0;
    
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations 
    I = uint8(255*I);
    
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        % add text annotation
        if tracks(detect_ind) ~= 0                     
            col = min(cols(tracks(detect_ind),:) + [.2 .2 .2],1);
            rloc = max(1,D(detect_ind).Centroid(2) - 30);
            cloc = max(1,D(detect_ind).Centroid(1) + 20);
            %I=trkRenderText(I,['id=' num2str(tracks(detect_ind))], floor(255*col), [rloc, cloc], 'bnd', 'left');
            I=trkRenderText(I,['id=' num2str(D(detect_ind).ID)], floor(255*col), [rloc, cloc], 'bnd', 'left');
            %I=trkRenderText(I,[num2str(round(D(detect_ind).TravelDistance)) 'px'], floor(255*col), [rloc+30, cloc], 'bnd', 'left');
            
            %I=trkRenderText(I,[num2str(D(detect_ind).Speed) 'px/f'], floor(255*col), [rloc+60, cloc], 'bnd', 'left');
            
        else
            h = text(D(detect_ind).Centroid(1), D(detect_ind).Centroid(2), 'false detection!');
            set(h, 'Color', [1 .4 .4] );
        end
    end

    % print the name of the experiment on top of the video
    I=trkRenderText(I,date_txt, [255 255 255], [10, 20], 'bnd', 'left');
    I=trkRenderText(I,num_txt, [255 255 255], [10, 180], 'bnd', 'left');
    I=trkRenderText(I,label_txt, [255 255 255], [10, 240], 'bnd', 'left');

    % show the image
    imshow(I);     pause(0.05);
    
    % store the image for writing a movie file
    mv{t} = I;
end


% put everything into a nice structure for the xml writer
Experiment = makeOutputStructure(D, Soma, Dlist, date_txt, label_txt, tracks, FrameMeasures, num_txt);


% write the xml file
xmlFileName = [folder num_txt '.xml'];
disp(['writing ' xmlFileName]);
trkWriteXMLFile(Experiment, xmlFileName);

% save the parameters in the experiment folder
disp(['writing to ' folder 'params.mat']);
save([folder 'params.mat'], 'FRANGI_THRESH', 'NUC_INT_THRESH', 'MIN_SIZE', 'SOMA_PERCENT_THRESH', 'W_THRESH', 'WT', 'WSH', 'WIN_SIZE', 'rmin', 'rmax');

% make a movie of the results
makemovie(mv, folder, [num_txt '.avi']);




%keyboard;







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




% ===================== SUPPORTING FUNCTIONS ==============================
%
%
% =========================================================================
function cols = color_list()

cols1 = summer(6);
cols1 = cols1(randperm(6),:);
cols2 = summer(8);
cols2 = cols2(randperm(8),:);
cols3 = summer(180);
cols3 = cols3(randperm(180),:);
cols = [cols1; cols2; cols3];



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

    Experiment.TimeStep(t).Time = t;
    Experiment.TimeStep(t).ImgDiff1 = FrameMeasures(t).ImgDiff1;
    Experiment.TimeStep(t).ImgDiff2 = FrameMeasures(t).ImgDiff2;
    Experiment.TimeStep(t).Entropy  = FrameMeasures(t).Entropy;
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




function A = make_adjacency(Dlist, WIN_SIZE, Ndetection)

A = zeros(Ndetection);
%A = sparse(A);

for t = 2:length(Dlist)
   for d = 1:length(Dlist{t})
        d_i = Dlist{t}(d);
        min_t = max(1, t-WIN_SIZE);
        for p = min_t:t-1
            A(d_i, Dlist{p}) = 1;            
        end
   end
end



function B1 = segment_nuclei2(I, J, THRESH, NUC_INT_THRESH, MIN_SIZE)


B = I <  THRESH; %-.0002;
B = bwareaopen(B, MIN_SIZE);
Bprop = regionprops(B, 'Eccentricity', 'PixelIdxList');
for i = 1:length(Bprop)
    if Bprop(i).Eccentricity > .85
        B(Bprop(i).PixelIdxList) = 0;
    end
end
%B(1:end,1) = 1; B(1:end,end) = 1; B(1,1:end) = 1; B(end,1:end) = 1;

%B = bwmorph(B, 'thicken', 3);    


B2 = J > NUC_INT_THRESH;
%B2 = bwareaopen(B2, MIN_SIZE, 4);

B1 = B | B2;

%keyboard


function M = segment_nuclei(I, NUC_INT_THRESH, MIN_SIZE)

%NUC_INT_THRESH = .5;
%NUC_INT_THRESH = .45;
%NUC_INT_THRESH = .52;
%NUC_INT_THRESH = .40;
%NUC_INT_THRESH = .36;
%NUC_INT_THRESH = .30;
%NUC_INT_THRESH = .28;
%NUC_INT_THRESH = .15;

%M = zeros(size(I));

M = I > NUC_INT_THRESH;

M = bwareaopen(M, MIN_SIZE, 8);

%keyboard;


function J = trkTo8Bits(I, lims)


%lims = stretchlim(I);         
J = imadjust(I, lims, []); 
J = uint8(J/2^8);



function makemovie(mv, folder, outputFilename)

disp('writing temporary image files');
for i = 1:length(mv)
   imwrite(mv{i}, [folder sprintf('%03d',i) '.png'], 'PNG'); 
end

%keyboard;

oldpath = pwd;

cd(folder);
cmd = ['mencoder "mf://*.png" -mf fps=10 -o ' outputFilename ' -ovc xvid -xvidencopts bitrate=3000'];
%cmd = ['ffmpeg -r 10 -b 600k -i %03d.png output.avi'];
disp(cmd);
system(cmd);
cd(oldpath);

% cmd = ['ffmpeg -r 10 -b 600k -i ' folder '%03d.png ' folder 'output.avi'];
% disp(cmd);
% system(cmd);

cmd = ['rm ' folder '*.png'];
disp(cmd);
system(cmd);



