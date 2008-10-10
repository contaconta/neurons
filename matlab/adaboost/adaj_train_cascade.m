%% load the parameters and path information
% ----------
vj_train_settings; 
versioninfo;


%% preparation

% prepare the log file
logfile(log_filenm, 'erase');                   % clear the log file
logfile(log_filenm, 'header', {appname, ['Version ' version], ['by ' author ', ' email], [num2str(TRAIN_POS) ' positive examples, ' num2str(TRAIN_NEG) ' negative examples.'], ['Started at ' datestr(now)],'-----------------------------------'});
logfile(log_filenm, 'column_labels', {'stage', 'step', 'Weak ID', 'polarity', 'theta', 'Di', 'Fi', 'di', 'fi', 'di(train)', 'fi(train)'});

% collect the training data into a struct
tic; disp('...collecting and processing the TRAIN data set.');
TRAIN = vj_collect_data(train1, train0, 'size', IMSIZE, 'normalize', NORM, 'data_limit', [TRAIN_POS TRAIN_NEG]);toc;

% collect a validation set
tic; disp('...collecting and processing the VALIDATION data set.');
VALIDATION = vj_collect_data(validation1, validation0, 'size', IMSIZE, 'normalize', NORM, 'data_limit', [TEST_POS, TEST_NEG]);toc;

% define a set of haar-like weak classifiers over the standard image size
tic; disp('...defining the haar-like weak classifiers.');
WEAK = vj_define_weak_classifiers(IMSIZE, 'types', [1 2 3 5]); toc; 
                    
%precompute the haar-like feature responses for each classifier over the
%training set and store them in a bigmatrix, PRE.
disp('...precomputing the haar-like feature responses of each classifier ');
disp(['   on the ' num2str(length(TRAIN)) ' training images (this may take quite some time).']);                        
PRE = vj_precompute_haar_response_new(TRAIN, WEAK, temp_filenm, temppath, []);


%% train the cascade

CASCADE = vj_cascade_init();    % initialize the CASCADE struct
i = 0;                          % cascade classifier index
Fi = 1;                         % current cascade false positive rate      
Di = 1;                         % current cascade detection rate


while (Fi > Ftarget)
    i = i + 1;
    disp(['============== NOW TRAINING CASCADE STAGE i = ' num2str(i) ' ==============']);
    ti = 0;        % the number of weak learners in current classifier

    if i == 1; Flast = 1; else Flast = prod([CASCADE(1:i-1).fi]); end
    if i == 1; Dlast = 1; else Dlast = prod([CASCADE(1:i-1).di]); end
     
    % the classifier must meet its detection rate and false positive rate 
    % goals before it is accepted as the next stage of the cascade
    while (Fi > fmax * Flast) || (Di < dmin * Dlast)
        ti = ti + 1;
        
        %% train the next weak classifier (for the strong classifier of stage i)
        disp('   ----------------------------------------------------------------------');
        disp(['...CASCADE stage ' num2str(i) ' training classifier hypothesis ti=' num2str(ti) '.']);
        if ti == 1
            CASCADE(i).CLASSIFIER = vj_adaboost(PRE, TRAIN, WEAK, ti);
        else
            CASCADE(i).CLASSIFIER = vj_adaboost(PRE, TRAIN, WEAK, ti, CASCADE(i).CLASSIFIER);
        end
        
        %% select the cascade threshold for stage i
        %  adjust the threshold for the current classifier until we find one
        %  which gives a satifactory detection rate (this changes the false alarm rate)
        [CASCADE, Fi, Di]  = vj_cascade_select_threshold(CASCADE, i, VALIDATION, dmin);

        % ====================  TEMPORARY  ==============================
        % to make sure we're actually improving on the training data
        gt = [TRAIN(:).class]';  C = zeros(size(gt));
        for j=1:length(TRAIN); 
            C(j) = vj_classify_cascade(CASCADE, TRAIN(j).II);
        end
        [tpr fpr FPs] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist');
        disp(['results on TRAIN data for CASCADE: Di=' num2str(tpr) ', (f)i=' num2str(fpr) ', #FPs = ' num2str(length(FPs)) ' (remember class 0 = FPs)' ]);               
        % ===============================================================
        
        % write training results to the log file
        logfile(log_filenm, 'write', [i ti CASCADE(i).CLASSIFIER.feature_index(ti) CASCADE(i).CLASSIFIER.polarity(ti) CASCADE(i).CLASSIFIER.theta(ti) Di Fi CASCADE(i).di CASCADE(i).fi tpr fpr]);
        
        % save the cascade to a file in case something bad happens and we need to restart
        save(cascade_filenm, 'CASCADE');
        disp(['...saved a temporary copy of CASCADE to ' cascade_filenm]);
        
        % hand-tuned stage goals (for first few cascade stages) loosly following Viola-Jones IJCV '04
        if i == 1; if (CASCADE(i).di > .99) && (CASCADE(i).fi < .50);   break; end; end;
        if i == 2; if (CASCADE(i).di > .99) && (CASCADE(i).fi < .40);   break; end; end;
        if i == 3; if (Di > dmin * Dlast) && (Fi < Flast*.35);  break; end; end;
        if i == 4; if (Di > dmin * Dlast) && (Fi < Flast*.35);  break; end; end;
    end
    
    %% prepare for the next stage of the cascade
    
    % recollect negative examples for the training and validation set which 
    % only include false positive examples from the current cascade (selected 
    % at random from full images)
    disp('...updating the TRAIN set with negative examples which cause false positives');
    TRAIN = vj_cascade_collect_data(train1, update0, TRAIN, CASCADE, 'size', IMSIZE, ...
                    'normalize', NORM, 'data_limit', [TRAIN_POS TRAIN_NEG]);
                
    disp('...updating the VALIDATION set with negative examples which cause false positives');
    VALIDATION = vj_cascade_collect_data(validation1, update0, VALIDATION, CASCADE, 'size', IMSIZE, ...
                    'normalize', NORM, 'data_limit', [TEST_POS TEST_NEG]);
     
    % we must precompute haar responses over all of the TRAIN set again
    disp(['...precomputing the haar-like feature responses of each classifier on the ' num2str(length(TRAIN)) ' training images (this may take quite some time).']);                       
    PRE = vj_precompute_haar_response_new(TRAIN, WEAK, temp_filenm, temppath, PRE);
end




