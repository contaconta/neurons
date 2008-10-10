%function ada_train_cascade( )


% parameters
% ----------
fmax = .3;          % maximum false positive rate for any classifier in the cascade
dmin = .99;         % minimum detection rate for any classifier in the cascade
Ftarget = 1e-5;     % Target false positive rate for the cascade
IMSIZE = [19 19];   % standard size for face images
TRAIN_POS = 6000;   % number of positive examples in the training set
TRAIN_NEG = 6000;   % number of negative examples in the training set
TEST_POS =  2000;   % number of positive examples in the test set
TEST_NEG  = 6000;   % number of negative examples in the test set

% path info
% ---------
temppath = [pwd '/mat/'];  temp_filenm = 'CASCADE_';
datapath = '/osshare/Work';
train1 = [datapath '/Data/face_databases/combined/train/face/'];
validation1 =  [datapath '/Data/face_databases/combined/test/face/'];
cascade_filenm = 'CASCADE_6000.mat';

% preparation
% -----------


% recollect negative examples for the training and validation set which only include false positive
% examples from the current cascade (selected at random from full images)
train0 = [datapath '/Data/face_databases/non-face_uncropped/'];
disp('...updating the TRAIN set with negative examples which cause false positives');
TRAIN = ada_cascade_collect_data(train1, train0, TRAIN, CASCADE, 'size', IMSIZE, ...
                'normalize', 1, 'data_limit', [TRAIN_POS TRAIN_NEG]);

validation0 = [datapath '/Data/face_databases/non-face_uncropped/'];
disp('...updating the VALIDATION set with negative examples which cause false positives');
VALIDATION = ada_cascade_collect_data(validation1, validation0, VALIDATION, CASCADE, 'size', IMSIZE, ...
                'normalize', 1, 'data_limit', [TEST_POS TEST_NEG]);

% unfortunately we must precompute haar responses over all of the TRAIN set again
disp(['...precomputing the haar-like feature responses of each classifier on the ' num2str(length(TRAIN)) ' training images (this may take quite some time).']);                       
PRE = ada_precompute_haar_response(TRAIN, WEAK, temp_filenm, temppath);

% define a set of haar-like weak classifiers over the standard image size
if ~ exist('WEAK', 'var');
    tic; disp('...defining the weak haar-like classifiers.');
    WEAK = ada_define_weak_classifiers(IMSIZE, 'types', [1 2 3 5]); toc; 
end

% train the cascade
% -----------------

CASCADE = CASCADE(1:length(CASCADE) -1);    % initialize the CASCADE struct
i = length(CASCADE)-1;                      % cascade classifier index
Fi = prod([CASCADE.fi]);                    % current cascade false positive rate      
Di = prod([CASCADE.di]);                    % current cascade detection rate


while (Fi > Ftarget)
    i = i + 1;
    disp(['============== NOW TRAINING CASCADE STAGE i = ' num2str(i) ' ==============']);
    ti = 0;         % the number of weak learners in current classifier
  
    if i == 1; Flast = 1; else Flast = prod([CASCADE(1:i-1).fi]); end
    if i == 1; Dlast = 1; else Dlast = prod([CASCADE(1:i-1).di]); end
    
    % the classifier must meet its detection rate and false positive rate 
    % goals before it is accepted as the next stage of the cascade
    while (Fi > fmax * Flast) || (Di < dmin * Dlast)
        ti = ti + 1;
        
        % train the next weak classifier for the strong classifier of stage i
        disp(['...CASCADE stage ' num2str(i) ' training classifier hypothesis ti=' num2str(ti) '.']);
        if ti == 1
            CASCADE(i).CLASSIFIER = ada_adaboost(PRE, TRAIN, WEAK, ti);
        else
            CASCADE(i).CLASSIFIER = ada_adaboost(PRE, TRAIN, WEAK, ti, CASCADE(i).CLASSIFIER);
        end
        
        % adjust the threshold for the current classifier until we find one
        % which gives a satifactory detection rate (this changes the false alarm rate)
        [CASCADE, Fi, Di]  = ada_cascade_select_threshold(CASCADE, i, VALIDATION, dmin);

        % save the cascade to a file in case something bad happens and we need to restart
        save(cascade_filenm, 'CASCADE');
        disp(['...saved a temporary copy of CASCADE to ' cascade_filenm]);
        
        % special hand-tuning of the first 7 stages of the cascade as in Viola-Jones IJCV '04
        if i == 1; if (CASCADE(i).di > .98) && (CASCADE(i).fi < .50);   break; end; end;
        if i == 2; if (CASCADE(i).di > .985) && (CASCADE(i).fi < .33);   break; end; end;
        if i == 3; if (Di > dmin * Dlast) && (Fi < Flast*.75);  break; end; end;
        if i == 4; if (Di > dmin * Dlast) && (Fi < Flast*.75);  break; end; end;
    end
       
    
    
    % recollect negative examples for the training and validation set which only include false positive
    % examples from the current cascade (selected at random from full images)
    train0 = [datapath '/Data/face_databases/non-face_uncropped/'];
    disp('...updating the TRAIN set with negative examples which cause false positives');
    TRAIN = ada_cascade_collect_data(train1, train0, TRAIN, CASCADE, 'size', IMSIZE, ...
                    'normalize', 1, 'data_limit', [TRAIN_POS TRAIN_NEG]);
                
    validation0 = [datapath '/Data/face_databases/non-face_uncropped/'];
    disp('...updating the VALIDATION set with negative examples which cause false positives');
    VALIDATION = ada_cascade_collect_data(validation1, validation0, VALIDATION, CASCADE, 'size', IMSIZE, ...
                    'normalize', 1, 'data_limit', [TEST_POS TEST_NEG]);
    
    % unfortunately we must precompute haar responses over all of the TRAIN set again
    disp(['...precomputing the haar-like feature responses of each classifier on the ' num2str(length(TRAIN)) ' training images (this may take quite some time).']);                       
    PRE = ada_precompute_haar_response(TRAIN, WEAK, temp_filenm, temppath);
end
