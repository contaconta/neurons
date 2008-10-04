%%-------------------------------------------------------------------------
%% PARAMETERS
%%-------------------------------------------------------------------------

fmax = .3;          % maximum false positive rate for any classifier in the cascade
dmin = .99;         % minimum detection rate for any classifier in the cascade
Ftarget = 1e-5;     % Target false positive rate for the cascade
IMSIZE = [24 24];   % standard size for face images
TRAIN_POS = 4000;   % number of positive examples in the training set
TRAIN_NEG = 4000;   % number of negative examples in the training set
TEST_POS =  4000;   % number of positive examples in the test set
TEST_NEG  = 4000;   % number of negative examples in the test set
NORM      = 1;      % normalize the variance of image intensity?
cascade_filenm = 'CASCADE_4000.mat';    % filename to store your learned cascade
log_filenm = 'CASCADE_4000.log';


%%-------------------------------------------------------------------------
%% PATH INFORMATION
%%-------------------------------------------------------------------------

% path and filename to temporary storage
temppath =      [pwd '/mat/'];  temp_filenm = 'BIGARRAY_';

% root path to where data resides
datapath =      '/osshare/Work';
train1 =        [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/train/face/'];
train0 =        [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/train/non-face/'];
validation1 =   [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/test/face/'];
validation0 =   [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/test/non-face/'];
update0 =       [datapath '/Data/face_databases/EPFL-CVLAB_faceDB/non-face_uncropped_images/'];
    