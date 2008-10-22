%-------------------------------------------------------------------------
% PARAMETERS
%-------------------------------------------------------------------------

IMSIZE      = [24 24];  % standard image size used for training & classification
fmax        = .3;      	% maximum false positive rate for any cascade stage
dmin        = .99;    	% minimum detection rate for any cascade stage
Ftarget     = 1e-5;     % target false positive rate for the entire cascade
TRAIN_POS   = 500;      % number of positive examples in the training set
TRAIN_NEG   = 500;      % number of negative examples in the training set
TEST_POS    = 500;      % number of positive examples in the test set
TEST_NEG    = 500;      % number of negative examples in the test set
NORM        = 1;        % normalize image intensity variance?

%-------------------------------------------------------------------------
% WEAK LEARNERS
%-------------------------------------------------------------------------

% list of the types of learners to use
learner_list = {'haar', 'spedge'};

% a list of the parameters passed to the weak learner definition functions, 
% in same order as learner_list.
learner_params = { {IMSIZE, 'type', {'haar1', 'haar2', 'haar3', 'haar5'}}, ...
                   {IMSIZE, 0:30:360-30} };

% % list of the types of learners to use
% learner_list = {'haar', 'haar', 'haar', 'haar'};
% 
% % a list of the parameters passed to the weak learner definition functions, 
% % in same order as learner_list.
% learner_params = { {IMSIZE, 'type', {'haar1'}}, ... 
%                    {IMSIZE, 'type', {'haar2'}}, ...
%                    {IMSIZE, 'type', {'haar3'}}, ...
%                    {IMSIZE, 'type', {'haar5'}}   };
                     

%-------------------------------------------------------------------------
% FILES & PATH INFORMATION
%-------------------------------------------------------------------------

% filename to store the cascaded classifier
cascade_filenm = 'CASCADE_500.mat';  
% filename to store the log file
log_filenm = 'CASCADE_500.log';         
% path and filename to temporary storage
temppath =      [pwd '/mat/'];  temp_filenm = 'BIGARRAY_';
% root path to where data resides
datapath =      '/osshare/Work';
% path to training set (+) class
train1 =        [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/train/face/'];
% path to training set (-) class
train0 =        [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/train/non-face/'];
% path to validation set (+) class
validation1 =   [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/test/face/'];
% path to validation set (-) class
validation0 =   [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/test/non-face/'];
% path to collection of images used to find new false positives for validation
update0 =       [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/non-face_uncropped_images/'];
% append the path to the spedges features 
path(path, [pwd '/../spedges/']);    