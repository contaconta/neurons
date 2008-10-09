%%-------------------------------------------------------------------------
%% PARAMETERS
%%-------------------------------------------------------------------------

fmax      = .3;    	% maximum false positive rate for any classifier in the cascade
dmin      = .99;   	% minimum detection rate for any classifier in the cascade
Ftarget   = 1e-5; 	% Target false positive rate for the cascade
IMSIZE    = [24 24];% standard size for face images
TRAIN_POS = 1688;   % number of positive examples in the training set
TRAIN_NEG = 1688;   % number of negative examples in the training set
TEST_POS  = 1688;   % number of positive examples in the test set
TEST_NEG  = 1688;   % number of negative examples in the test set
NORM      = 0;      % normalize the variance of image intensity?
cascade_filenm = 'NUCLEUS_CASCADE_1688.mat';
log_filenm = 'NUCLEUS_CASCADE_1688.log';

%%-------------------------------------------------------------------------
%% PATH INFORMATION
%%-------------------------------------------------------------------------

temppath =      [pwd '/mat/'];  temp_filenm = 'NUCLEUS_CASCADE_';
datapath =      '/osshare/Work';
% train1 =        [datapath '/Data/nucleus_training24x24/train/pos/'];
% train0 =        [datapath '/Data/nucleus_training24x24/train/neg/'];
% validation1 =   [datapath '/Data/nucleus_training24x24/test/pos/'];
% validation0 =   [datapath '/Data/nucleus_training24x24/test/neg/'];
% update0 =       [datapath '/Data/nucleus_training24x24/mips/'];

train1 =        [datapath '/Data/nucleus_training_rotated/train/pos/'];
train0 =        [datapath '/Data/nucleus_training_rotated/train/neg/'];
validation1 =   [datapath '/Data/nucleus_training_rotated/test/pos/'];
validation0 =   [datapath '/Data/nucleus_training_rotated/test/neg/'];
update0 =       [datapath '/Data/nucleus_training_rotated/mips/'];