function Green =  trkTrackCellsAndWriteToViper(folder, resultsFolder, SeqIndexStr)

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
[Cells, ~, trkSeq, timeSeq] = trkGenerateNucleiGraphAndTrack(CellsList, Cells, WIN_SIZE, WT, WSH, W_THRESH, MIN_TRACK_LENGTH, SPATIAL_DIST_THRESH);
toc 

numberofTracks = 0;
for i =1:length(timeSeq)
    if ~isempty(trkSeq{i})
        numberofTracks = numberofTracks + 1;
    end
end

%% Generate the xml structure and write
ST = xml2struct('sampleViperCircleTracking.xml');
outputAnnotFile = [resultsFolder SeqIndexStr];

ST.viper.data.sourcefile.Attributes.filename = ['./' SeqIndexStr 'Original.mpg'];

ST.viper.data.sourcefile.object = cell(1, numberofTracks);

trackIdx = 1;

for i = 1:length(trkSeq)
    if ~isempty(trkSeq{i})
        % Get the framespan
        listOfStarts = timeSeq{i}(1);
        listOfEnds = [];
        for k =2:length(timeSeq{i})
            if timeSeq{i}(k)-1 > timeSeq{i}(k-1)
                listOfEnds   = [listOfEnds timeSeq{i}(k-1)];%#ok
                listOfStarts = [listOfStarts timeSeq{i}(k)];%#ok
            end
        end
        listOfEnds = [listOfEnds timeSeq{i}(k)];%#ok
        if length(listOfStarts) ~= length(listOfEnds)
           error('thos 2 must have the same size');
        end
        framespanString = '';
        for k = 1:length(listOfStarts)-1
            framespanString = [framespanString int2str(listOfStarts(k)) ':' int2str(listOfEnds(k)) ' '];%#ok
        end
        k = length(listOfStarts);
        framespanString = [framespanString int2str(listOfStarts(k)) ':' int2str(listOfEnds(k))];%#ok
        % Get the framespan DONE !!
        ST.viper.data.sourcefile.object{trackIdx}.Attributes.framespan        = framespanString;
        ST.viper.data.sourcefile.object{trackIdx}.Attributes.name             = 'Cell';
        ST.viper.data.sourcefile.object{trackIdx}.Attributes.id               = num2str(trackIdx-1);
        ST.viper.data.sourcefile.object{trackIdx}.attribute.Attributes.name   = 'location';
        ST.viper.data.sourcefile.object{trackIdx}.attribute.data_colon_circle = cell(1, length(timeSeq{i}));
        
        for k =1:length(timeSeq{i})
            currentCell = Cells(trkSeq{i}(k));
            
            ST.viper.data.sourcefile.object{trackIdx}.attribute.data_colon_circle{k}.Text = '';
            ST.viper.data.sourcefile.object{trackIdx}.attribute.data_colon_circle{k}.Attributes.framespan = [int2str(currentCell.Time) ':' int2str(currentCell.Time)];
            ST.viper.data.sourcefile.object{trackIdx}.attribute.data_colon_circle{k}.Attributes.radius    = num2str(round(currentCell.NucleusMinorAxisLength / 2.0));
            ST.viper.data.sourcefile.object{trackIdx}.attribute.data_colon_circle{k}.Attributes.x         = num2str(round(currentCell.NucleusCentroid(1))-1);
            ST.viper.data.sourcefile.object{trackIdx}.attribute.data_colon_circle{k}.Attributes.y         = num2str(round(currentCell.NucleusCentroid(2))-1);
        end
    	trackIdx= trackIdx + 1;
    end
end

% 
% 
% 
% ST.viper.config.descriptor{2}.attribute{3} = ST.viper.config.descriptor{2}.attribute{2};
% ST.viper.config.descriptor{2}.attribute{2} = ST.viper.config.descriptor{2}.attribute{1};
% ST.viper.config.descriptor{2}.attribute{1} = [];
% ST.viper.config.descriptor{2}.attribute{1}.Attributes.dynamic = 'false';
% ST.viper.config.descriptor{2}.attribute{1}.Attributes.name = 'CellId';
% ST.viper.config.descriptor{2}.attribute{1}.Attributes.type = 'http://lamp.cfar.umd.edu/viperdata#svalue';
% ST.viper.config.descriptor{2}.attribute{1}.Text = '';
% 
% ST.viper.data.sourcefile.object = cell(1, 1);
% trackIdx = 1;
% for i =1:1
%     
%     if ~isempty(trkSeq{i})
%         
%         % Get the framespan
%         listOfStarts = timeSeq{i}(1);
%         listOfEnds = [];
%         for k =2:length(timeSeq{i})
%             if timeSeq{i}(k)-1 > timeSeq{i}(k-1)
%                 listOfEnds   = [listOfEnds timeSeq{i}(k-1)];%#ok
%                 listOfStarts = [listOfStarts timeSeq{i}(k)];%#ok
%             end
%         end
%         listOfEnds = [listOfEnds timeSeq{i}(k)];%#ok
%         if length(listOfStarts) ~= length(listOfEnds)
%            error('thos 2 must have the same size');
%         end
%         framespanString = '';
%         for k = 1:length(listOfStarts)-1
%             framespanString = [framespanString int2str(listOfStarts(k)) ':' int2str(listOfEnds(k)) ' '];%#ok
%         end
%         k = length(listOfStarts);
%         framespanString = [framespanString int2str(listOfStarts(k)) ':' int2str(listOfEnds(k))];%#ok
%         % Get the framespan DONE !!
%         ST.viper.data.sourcefile.object{trackIdx}.Attributes.framespan  = framespanString;
%         ST.viper.data.sourcefile.object{trackIdx}.Attributes.name       = 'Cell';
%         ST.viper.data.sourcefile.object{trackIdx}.Attributes.id         = num2str(trackIdx-1);
%                 
%         ST.viper.data.sourcefile.object{trackIdx}.attribute = cell(1, 3);
%         
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{1}.Attributes.name = 'CellId';
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{1}.data_colon_svalue.Text = '';
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{1}.data_colon_svalue.Attributes.value = num2str(trackIdx);
%         
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.Attributes.name = 'Nucleus';
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.data_colon_polygon = cell(1, length(trkSeq{i}));
%         
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.Attributes.name = 'Soma';
%         ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.data_colon_polygon = cell(1, length(trkSeq{i}));
%         
%         % done with the track
%         for k =1:length(trkSeq{i})
%             currentCell = Cells(trkSeq{i}(k));
%             
%             % Nucleus
%             ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.data_colon_polygon{k}.Attribute.framespan = [int2str(currentCell.Time) ':' int2str(currentCell.Time)];
%             
%             listOfSpaceIdx = 1:3:length(currentCell.NucleusListBoundaryPointsY);
%             ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.data_colon_polygon{k}.data_colon_point = cell(1, length(listOfSpaceIdx));
%             for w = 1:length(listOfSpaceIdx)
%                 ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.data_colon_polygon{k}.data_colon_point{w}.Text = '';
%                 ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.data_colon_polygon{k}.data_colon_point{w}.Attributes.y = int2str(currentCell.NucleusListBoundaryPointsX(listOfSpaceIdx(w))-1);
%                 ST.viper.data.sourcefile.object{trackIdx}.attribute{2}.data_colon_polygon{k}.data_colon_point{w}.Attributes.x = int2str(currentCell.NucleusListBoundaryPointsY(listOfSpaceIdx(w))-1);
%             end
%             
%             % Soma
%             ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.data_colon_polygon{k}.Attribute.framespan = [int2str(currentCell.Time) ':' int2str(currentCell.Time)];
%             listOfSpaceIdx = 1:3:length(currentCell.SomaListBoundaryPointsY);
%             ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.data_colon_polygon{k}.data_colon_point = cell(1, length(listOfSpaceIdx));
%             for w = 1:length(listOfSpaceIdx)
%                 ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.data_colon_polygon{k}.data_colon_point{w}.Text = '';
%                 ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.data_colon_polygon{k}.data_colon_point{w}.Attributes.y = int2str(currentCell.SomaListBoundaryPointsX(listOfSpaceIdx(w))-1);
%                 ST.viper.data.sourcefile.object{trackIdx}.attribute{3}.data_colon_polygon{k}.data_colon_point{w}.Attributes.x = int2str(currentCell.SomaListBoundaryPointsY(listOfSpaceIdx(w))-1);
%             end
%         end
%         
%         
%         trackIdx= trackIdx + 1;
%     end
% end

disp('done with the structure !!')

struct2xml(ST, outputAnnotFile);

