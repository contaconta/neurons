%-------------------------------------------------------------------------
% PARAMETERS
%-------------------------------------------------------------------------

IMSIZE          = [24 24];  % standard image size used for training & classification
fmax            = .3;      	% maximum false positive rate for any cascade stage
dmin            = .99;    	% minimum detection rate for any cascade stage
Ftarget         = 1e-5;     % target false positive rate for the entire cascade
TRAIN_POS       = 1000;     % number of positive examples in the training set
TRAIN_NEG       = 1000;     % number of negative examples in the training set
VALIDATION_POS  = 1000;     % number of positive examples in the validation set
VALIDATION_NEG  = 1000;     % number of negative examples in the validation set
NORM            = 1;        % normalize image intensity variance? (FACES 1, NUCLEI 0)

rand('twister', 100);       % seed the random variable

%-------------------------------------------------------------------------
% FILES & PATH INFORMATION
%-------------------------------------------------------------------------

FILES.datestr         = datestr(now, 'dd-mmm-yyyy-HH.MM.SS');
FILES.cascade_filenm  = ['SPDIFF' FILES.datestr '.mat'];   % filename to store the cascaded classifier
FILES.log_filenm      = ['SPDIFF' FILES.datestr '.log'];   % filename to store the log file    
FILES.temppath        = [pwd '/mat/'];                      % temporary storage path
FILES.train_filenm    = [pwd '/mat/TRAIN_FEATURESspdiff.dat'];    % temporary storage filename
FILES.valid_filenm    = [pwd '/mat/VALID_FEATURESspdiff.dat'];
FILES.memory          = 300000000;                          % size of memory to use in bytes for each bigmatrix 
FILES.precision       = 'single';                           % precision of stored values
path(path, [pwd '/../spedges/']);                           % append the path to the spedges features 
path(path, [pwd '/../toolboxes/bigmatrix/']);               % append the path to bigmatrix

%-------------------------------------------------------------------------
% WEAK LEARNERS
%-------------------------------------------------------------------------
LEARNERS = [];


LEARNERS(length(LEARNERS)+1).feature_type   = 'intmean';
LEARNERS(length(LEARNERS)).IMSIZE        	= IMSIZE;

LEARNERS(length(LEARNERS)+1).feature_type 	= 'intvar';
LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;

% LEARNERS(length(LEARNERS)+1).feature_type 	= 'haar';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).shapes           = {'vert2', 'horz2', 'vert3', 'checker'};

% LEARNERS(length(LEARNERS)+1).feature_type   = 'spedge';
% LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
% LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
% LEARNERS(length(LEARNERS)).sigma            = [1 1.5 2 3];  %2;

LEARNERS(length(LEARNERS)+1).feature_type   = 'spdiff';
LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
LEARNERS(length(LEARNERS)).angles           = 0:30:360-30;
LEARNERS(length(LEARNERS)).sigma            = [1 2];  %[1 1.5 2 3];  %2;

%-------------------------------------------------------------------------
% TRAINING & VALIDATION DATA SETS
%-------------------------------------------------------------------------

DATASETS.filelist = 'nuclei-rotated.txt';       % file containing list of training images
%DATASETS.filelist = 'faces.txt';                % file containing list of training images

% parameters for updating the negative examples
DATASETS.delta          = 10;                  % detector step size
DATASETS.scale_limits   = [1 2];               % detector scales (use if objects only exist at certain scales)


DATASETS.IMSIZE = IMSIZE; DATASETS.NORMALIZE = NORM;
DATASETS.TRAIN_POS = TRAIN_POS; DATASETS.TRAIN_NEG = TRAIN_NEG;
DATASETS.VALIDATION_POS = VALIDATION_POS; DATASETS.VALIDATION_NEG = VALIDATION_NEG;


