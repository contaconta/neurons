%-------------------------------------------------------------------------
% PARAMETERS
%-------------------------------------------------------------------------

Ftarget         = 1e-5;     % target false positive rate for the entire cascade
Dtarget         = .90;      % target detection rate for the entire cascade
TRAIN_POS       = 1000;     % number of positive examples in the training set
TRAIN_NEG       = 1000;     % number of negative examples in the training set
VALIDATION_POS  = 1000;     % number of positive examples in the validation set
VALIDATION_NEG  = 1000;     % number of negative examples in the validation set
NORM            = 1;        % normalize intensity? (1=FACES,NUCLEI,PERSONS, 0=MITO,CONTOURS)

rand('twister', 100);       % seed the random variable

%-------------------------------------------------------------------------
% FILES & PATH INFORMATION
%-------------------------------------------------------------------------

FILES.datestr         = datestr(now, 'dd-mmm-yyyy-HH.MM.SS');
FILES.cascade_filenm  = ['SPmix_repeat_prob' FILES.datestr '.mat'];   % filename to store the cascaded classifier
FILES.log_filenm      = ['SPmix_repeat_prob' FILES.datestr '.log'];   % filename to store the log file    
FILES.temppath        = [pwd '/mat/'];                      % temporary storage path
FILES.train_filenm    = [pwd '/mat/TRAIN_FEATURES.dat'];    % temporary storage filename
FILES.valid_filenm    = [pwd '/mat/VALID_FEATURES.dat'];
FILES.memory          = 300000000;                          % size of memory to use in bytes for each bigmatrix 
FILES.precision       = 'single';                           % precision of stored values
path(path, [pwd '/../spedges/']);                           % append the path to the spedges features 
path(path, [pwd '/../toolboxes/bigmatrix/']);               % append the path to bigmatrix
path(path, [pwd '/../toolboxes/kevin/']);                   % append the path to kevin

%-------------------------------------------------------------------------
% TRAINING & VALIDATION DATA SETS
%-------------------------------------------------------------------------

%DATASETS.filelist = 'nuclei-rotated.txt';   DATASETS.scale_limits = [.6 2]; IMSIZE = [24 24];      
%DATASETS.filelist = 'faces.txt';            DATASETS.scale_limits = [.6 5]; IMSIZE = [24 24];
%DATASETS.filelist = 'mitochondria48.txt';   DATASETS.scale_limits = [2 9];  IMSIZE = [24 24];   
%DATASETS.filelist = 'mitochondria24.txt';   DATASETS.scale_limits = [2 9];  IMSIZE = [24 24];
%DATASETS.filelist = 'nuclei24.txt';         DATASETS.scale_limits = [.62];  IMSIZE = [24 24];
%DATASETS.filelist = 'contours24.txt';       DATASETS.scale_limits = [1];    IMSIZE = [24 24]; 
DATASETS.filelist = 'persons24x64.txt';     DATASETS.scale_limits = [1 5];  IMSIZE = [64 24];

% parameters for updating the negative examples
DATASETS.delta          = 10;       % detector step size

DATASETS.IMSIZE = IMSIZE; DATASETS.NORMALIZE = NORM;
DATASETS.TRAIN_POS = TRAIN_POS; DATASETS.TRAIN_NEG = TRAIN_NEG;
DATASETS.VALIDATION_POS = VALIDATION_POS; DATASETS.VALIDATION_NEG = VALIDATION_NEG;

%-------------------------------------------------------------------------
% WEAK LEARNERS
%-------------------------------------------------------------------------
LEARNERS = [];

LEARNERS(length(LEARNERS)+1).feature_type   = 'intmean';
LEARNERS(length(LEARNERS)).IMSIZE        	= IMSIZE;

LEARNERS(length(LEARNERS)+1).feature_type 	= 'intvar';
LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;

LEARNERS(length(LEARNERS)+1).feature_type 	= 'haar';
LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
LEARNERS(length(LEARNERS)).shapes           = {'vert2', 'horz2', 'vert3', 'checker'};
LEARNERS(length(LEARNERS)).SCAN_Y_STEP      = 3;  % [3 persons, 1 all others]
LEARNERS(length(LEARNERS)).SCAN_X_STEP      = 1;  

% LEARNERS(length(LEARNERS)+1).feature_type   = 'spedge';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).stride           = 2; 
% LEARNERS(length(LEARNERS)).edge_methods     = [1 2 3 4 5 6];

% LEARNERS(length(LEARNERS)+1).feature_type   = 'spdiff';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).stride           = 2; 
% LEARNERS(length(LEARNERS)).edge_methods     = [23:27];

% LEARNERS(length(LEARNERS)+1).feature_type   = 'hog';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).bins             = 9;
% LEARNERS(length(LEARNERS)).cellsize         = [4 4];
% LEARNERS(length(LEARNERS)).blocksize        = [2 2];

