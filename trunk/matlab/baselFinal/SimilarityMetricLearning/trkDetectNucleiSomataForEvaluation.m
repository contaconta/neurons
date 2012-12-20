function [Nuclei, Somata] =  trkDetectNucleiSomataForEvaluation(folder, magnification)
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
[Red  , ~]   = trkReadImagesAndNormalize(TMAX, Rfolder);
[Green, ~] = trkReadImagesAndNormalize(TMAX, Gfolder);

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