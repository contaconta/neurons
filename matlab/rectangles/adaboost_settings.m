%% PARAMETERS

N_features = 1000;      % # of features to consider each boosting round
N_pos = 5000;           % # of requested positive training examples
N_total = 50000;        % # of total training examples
N_SAMPLES = 20000;      % # of negative examples to use when choosing optimal learner parameters
T = 5000;               % maximum rounds of boosting

IMSIZE = [24 24];       % size of the classification window

% folders containing the data sets
DATA_FOLDER = '/osshare/Work/Data/face_databases/EPFL-CVLAB_faceDB/';
pos_train_folder = [DATA FOLDER 'train/pos/'];
neg_train_folder = [DATA FOLDER 'non-face_uncropped_images/'];
pos_valid_folder = [DATA FOLDER 'test/pos/'];
neg_valid_folder = [DATA FOLDER 'non-face_uncropped_images/'];


%% COMPILE ANY MISSING MEX FILES

if ~exist(['weight_sample_mex.' mexext], 'file')
     compile_weight_sample;
end
