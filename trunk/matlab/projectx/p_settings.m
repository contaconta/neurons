
FILES.NAME = ['Test' '_'];          % descriptive prefix string to identify experiment files 

%-------------------------------------------------------------------------
% TRAINING PARAMETERS
%-------------------------------------------------------------------------

targetF                 = 1e-5;     % target false positive rate for the entire cascade
targetD                 = .90;      % target detection rate for the entire cascade
Nstages                 = 15;       % number of cascade stages
DATASETS.TRAIN_POS      = 500;      % number of positive examples in the training set
DATASETS.TRAIN_NEG      = 500;      % number of negative examples in the training set
DATASETS.VALIDATION_POS = 500;      % number of positive examples in the validation set
DATASETS.VALIDATION_NEG = 500;      % number of negative examples in the validation set
DATASETS.NORM           = 1;        % normalize intensity? (1=FACES,NUCLEI,PERSONS, 0=MITO,CONTOURS)

rand('twister', 100);      % seed the random variable

%-------------------------------------------------------------------------
% FILES & PATH INFORMATION
%-------------------------------------------------------------------------

FILES.datestr         = datestr(now, 'mmmddyyyy-HHMMSS');  %datestr(now, 'dd-mmm-yyyy-HH.MM.SS');
FILES.computername    = 'calcifer';                         % computer experiment was run on
FILES.cascade_filenm  = [FILES.NAME FILES.datestr FILES.computername '.mat'];   % filename to store the cascaded classifier
FILES.log_filenm      = ['./logs/' FILES.NAME FILES.datestr FILES.computername '.log'];   % filename to store the log file    
path(path, [pwd '/../spedges/']);                           % append the path to the ray's toolbox 
path(path, [pwd '/../toolboxes/kevin/']);                   % append the path to kevin's toolbox
path(path, [pwd '/bin/']);                                  % append the path to sub-functions

%-------------------------------------------------------------------------
% DATA SETS FOR TRAINING & VALIDATION 
%-------------------------------------------------------------------------

%DATASETS.filelist = 'nuclei-rotated.txt';   DATASETS.scale_limits = [.6 2]; DATASETS.IMSIZE = [24 24];      
%DATASETS.filelist = 'faces.txt';            DATASETS.scale_limits = [.6 5]; DATASETS.IMSIZE = [24 24];
%DATASETS.filelist = 'mitochondria48.txt';   DATASETS.scale_limits = [2 9];  DATASETS.IMSIZE = [24 24];   
%DATASETS.filelist = 'mitochondria24.txt';   DATASETS.scale_limits = [2 9];  DATASETS.IMSIZE = [24 24];
DATASETS.filelist = 'nuclei24.txt';         DATASETS.scale_limits = [1 3.3]; DATASETS.IMSIZE = [24 24];
%DATASETS.filelist = 'persons24x64.txt';     DATASETS.scale_limits = [1 5];  DATASETS.IMSIZE = [64 24];
%DATASETS.filelist = 'persons48x128.txt';    DATASETS.scale_limits = [1 5];  DATASETS.IMSIZE = [128 48];

% parameters for updating the negative examples
DATASETS.delta          = 10;       % re-collection scan step size

%-------------------------------------------------------------------------
% WEAK LEARNERS
%-------------------------------------------------------------------------
LEARNERS = [];

LEARNERS(length(LEARNERS)+1).feature_type   = 'intmean';
LEARNERS(length(LEARNERS)).IMSIZE        	= DATASETS.IMSIZE;

LEARNERS(length(LEARNERS)+1).feature_type 	= 'intvar';
LEARNERS(length(LEARNERS)).IMSIZE           = DATASETS.IMSIZE;

% LEARNERS(length(LEARNERS)+1).feature_type 	= 'haar';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).shapes           = {'vert2', 'horz2', 'vert3', 'checker'};
% LEARNERS(length(LEARNERS)).SCAN_Y_STEP      = 1;  % [6 persons, 1 all others]
% LEARNERS(length(LEARNERS)).SCAN_X_STEP      = 1;  % [2 persons, 1 all others]

% LEARNERS(length(LEARNERS)+1).feature_type   = 'spedge';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).stride           = 2; 
% LEARNERS(length(LEARNERS)).edge_methods     = [11:15];
% 
% LEARNERS(length(LEARNERS)+1).feature_type   = 'spdiff';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).stride           = 2;    % normally 2, 3 for persons.
% LEARNERS(length(LEARNERS)).edge_methods     = 11:15;% mix = [11 13 15 23 25 27 28];  canny=[11:15];  sobel=[23:28];
% 
% LEARNERS(length(LEARNERS)+1).feature_type   = 'spangle';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).stride           = 2; 
% LEARNERS(length(LEARNERS)).edge_methods     = 11:15;
% 
% LEARNERS(length(LEARNERS)+1).feature_type   = 'spnorm';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).stride           = 2; 
% LEARNERS(length(LEARNERS)).edge_methods     = 11:15;

LEARNERS(length(LEARNERS)+1).feature_type   = 'hog';
LEARNERS(length(LEARNERS)).IMSIZE           = DATASETS.IMSIZE;
LEARNERS(length(LEARNERS)).bins             = 9;
LEARNERS(length(LEARNERS)).cellsize         = [4 4];   % [8 8] for persons
LEARNERS(length(LEARNERS)).blocksize        = [2 2];

