%-------------------------------------------------------------------------
% PARAMETERS
%-------------------------------------------------------------------------

fmax = .3;          % maximum false positive rate for any classifier in the cascade
dmin = .99;         % minimum detection rate for any classifier in the cascade
Ftarget = 1e-5;     % Target false positive rate for the cascade
IMSIZE = [24 24];   % standard size for face images
TRAIN_POS = 500;    % number of positive examples in the training set
TRAIN_NEG = 500;    % number of negative examples in the training set
TEST_POS =  500;    % number of positive examples in the test set
TEST_NEG  = 500;    % number of negative examples in the test set
NORM      = 1;      % normalize the variance of image intensity?


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
    