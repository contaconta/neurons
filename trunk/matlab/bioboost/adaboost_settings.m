%% PARAMETERS

EXP_NAME = 'TEST';          % name of experiment 

N_features = 2000;          % # of features to consider each boosting round
N_pos = 3000;               % # of requested positive training examples
N_total = 200000;           % # of total training examples
N_SAMPLES = 25000;          % # of negative examples to use when choosing optimal learner parameters
T = 2000;                   % maximum rounds of boosting
EVAL = 1;                   % 1 = evaluate every boosting round/ 0 = no evaluation
SE = 'francois';               % example sampling method: 'kevin' or 'francois'


                            

DATASET = 'D2';        % the data set to use.  'D', 'Dplus40', ...


[s,host] = system('hostname'); host = strtrim(host);  	% computer hostname
date = datestr(now, 'mmmddyyyy-HHMMSS');                % the current date & time

%% folders containing the data sets
%DATA_FOLDER = '/osshare/Work/Data/face_databases/EPFL-CVLAB_faceDB/';
%pos_train_folder = [DATA_FOLDER 'train/pos/'];
%neg_train_folder = [DATA_FOLDER 'non-face_uncropped_images/'];
%pos_valid_folder = [DATA_FOLDER 'test/pos/'];
%neg_valid_folder = [DATA_FOLDER 'non-face_uncropped_images/'];

results_folder = [pwd '/results/'];
if ~exist(results_folder, 'dir'); mkdir(results_folder); end;


% compile any missing files
adaboost_compile_mex;


% list of T values to evaluate performance on
EVALUATE_LIST = [50 100 200 400 600 800 1000 1200 1400 1600 1800 2000];