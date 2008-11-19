%-------------------------------------------------------------------------
% PARAMETERS
%-------------------------------------------------------------------------

IMSIZE          = [24 24];  % standard image size used for training & classification
fmax            = .3;      	% maximum false positive rate for any cascade stage
dmin            = .99;    	% minimum detection rate for any cascade stage
Ftarget         = 1e-5;     % target false positive rate for the entire cascade
TRAIN_POS       = 500;     % number of positive examples in the training set
TRAIN_NEG       = 500;     % number of negative examples in the training set
VALIDATION_POS  = 500;     % number of positive examples in the validation set
VALIDATION_NEG  = 500;     % number of negative examples in the validation set
NORM            = 0;        % normalize image intensity variance?

rand('twister', 100);       % seed the random variable

%-------------------------------------------------------------------------
% FILES & PATH INFORMATION
%-------------------------------------------------------------------------

FILES.cascade_filenm  = 'CASCADE_SPEDGE1000.mat';           % filename to store the cascaded classifier
FILES.log_filenm      = 'CASCADE_SPEDGE1000.log';           % filename to store the log file    
FILES.temppath        = [pwd '/mat/'];                      % temporary storage path
FILES.train_filenm    = [pwd '/mat/TRAIN_FEATURES.dat'];    % temporary storage filename
FILES.valid_filenm    = [pwd '/mat/VALID_FEATURES.dat'];
FILES.memory          = 100000000;                          % size of memory to use in bytes for each bigmatrix 
path(path, [pwd '/../spedges/']);       % append the path to the spedges features 

%-------------------------------------------------------------------------
% WEAK LEARNERS
%-------------------------------------------------------------------------

LEARNERS(1).feature_type    = 'haar';
LEARNERS(1).IMSIZE          = IMSIZE;
LEARNERS(1).shapes          = {'vert2', 'horz2', 'vert3', 'checker'};

LEARNERS(2).feature_type    = 'spedge';
LEARNERS(2).IMSIZE          = IMSIZE;
LEARNERS(2).angles          = 0:30:360-30;
LEARNERS(2).sigma           = [1 1.5 2 3];  %2;

%-------------------------------------------------------------------------
% TRAINING & VALIDATION DATA SETS
%-------------------------------------------------------------------------

DATASETS.filelist = 'nuclei-rotated.txt';       % file containing list of training images

% parameters for updating the negative examples
DATASETS.delta          = 100;                  % detector step size
DATASETS.scale_limits   = [.5 2];               % detector scales (use if objects only exist at certain scales)


DATASETS.IMSIZE = IMSIZE; DATASETS.NORMALIZE = NORM;
DATASETS.TRAIN_POS = TRAIN_POS; DATASETS.TRAIN_NEG = TRAIN_NEG;
DATASETS.VALIDATION_POS = VALIDATION_POS; DATASETS.VALIDATION_NEG = VALIDATION_NEG;








% % path to training set (+) class
% train1 =        [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/train/face/'];
% % path to training set (-) class
% train0 =        [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/train/non-face/'];
% % path to validation set (+) class
% validation1 =   [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/test/face/'];
% % path to validation set (-) class
% validation0 =   [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/test/non-face/'];
% % path to collection of images used to find new false positives for validation
% update0 =       [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/non-face_uncropped_images/'];
