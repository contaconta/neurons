%% PARAMETERS

EXP_NAME = 'TEST';          % name of experiment 

N_features = 2000;          % # of features to consider each boosting round
N_pos = 5000;               % # of requested positive training examples
N_total = 200000;           % # of total training examples
N_SAMPLES = 25000;          % # of negative examples to use when choosing optimal learner parameters
T = 20;                   % maximum rounds of boosting
CONNECTEDNESS = 0.8;        % probability rectangles will be connected
EVAL = 0;                   % 1 = evaluate every boosting round/ 0 = no evaluation
SE = 'francois';               % example sampling method: 'kevin' or 'francois'

RANK = 4;                   % rectangle complexity
NORM = 'DNORM';             % ANORM = all normalized, DNORM = dynamic, NONORM = no normalize
RectMethod = 'Viola-Jones';      % shape generation method 'Viola-Jones', 'Karim1',
mixrate = 0.25;             %'Simple', 'Kevin', 'VJSPECIAL', '33', '50',
                            %'Asymmetric-Mix', 'rank-fixed', 'Lienhart',
                            %'LienhardNO3', 'lisymm', 'liasymm', 'ramirez'

IMSIZE = [24 24];           % size of the classification window

%host = hostname;                                       % compute hostname
[s,host] = system('hostname'); host = strtrim(host);  	% computer hostname
date = datestr(now, 'mmmddyyyy-HHMMSS');                % the current date & time

%% folders containing the data sets
DATA_FOLDER = '/osshare/Work/Data/face_databases/EPFL-CVLAB_faceDB/';
pos_train_folder = [DATA_FOLDER 'train/pos/'];
neg_train_folder = [DATA_FOLDER 'non-face_uncropped_images/'];
pos_valid_folder = [DATA_FOLDER 'test/pos/'];
neg_valid_folder = [DATA_FOLDER 'non-face_uncropped_images/'];

results_folder = [pwd '/results/'];
if ~exist(results_folder, 'dir'); mkdir(results_folder); end;


% compile any missing files
compile_mex_files;

% fix random number generator (disable for real experiments)
%load randomState.mat;
%defaultStream.State = randomState;