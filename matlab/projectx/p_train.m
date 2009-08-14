%% P_TRAIN trains a strong classifier for use in a detector from weak learners
%
%   Returns CASCADE, a structure containing a boosting-based classifier for
%   detecting objects in images.
%
%   See also P_SETTINGS

%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%               Aurelien Lucchi     http://cvlab.epfl.ch/~lucchi/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

%% ============================== load parameters and path information
%% ============================================
p_settings;     % load settings from file
p_versioninfo;  % load version info from file

% to save LabelMe indexing time, we check to see if we've already indexed
DATASETS = load_labelme_index(DATASETS);

%% ============================== preparation =====================================================================

% define the performance goals for each stage
BOOST = p_stage_goals(BOOST);

% initialize the log file
logfile(EXPERIMENT.log_filenm, 'erase');logfile(EXPERIMENT.log_filenm, 'header', {INFO.appname, ['Version ' INFO.version], ['by ' INFO.author ', ' INFO.email], [num2str(DATASETS.TRAIN_POS) ' positive examples, ' num2str(DATASETS.TRAIN_NEG) ' negative examples.'], ['DATASETS from ' DATASETS.labelme_pos_query], ['LEARNERS ' strcat(LEARNERS.types{:})],['Started at ' datestr(now)], INFO.copyright, '-----------------------------------'});
logfile(EXPERIMENT.log_filenm, 'column_labels', {'stage', 'step', 'Weak ID', 'Di', 'Fi', 'di', 'fi', 'di(train)', 'fi(train)', 'FPs', 'LEARNER'});

% ask the user if they'd like to precompute feature responses 
%if strcmp(input('\nPrecompute feature responses to speed up training time (y/n)? ', 's'), 'y'); DATASETS.precomputed = 1; else; DATASETS.precomputed = 0; end; %#ok<NOSEM>

% define the weak learners
LEARNERS = p_EnumerateLearners(LEARNERS, DATASETS.IMSIZE);

% collect the training data set
tic; disp(['...collecting the ' num2str(DATASETS.TRAIN_POS + DATASETS.TRAIN_NEG) ' examples in the TRAIN set.']);
[TRAIN, DATASETS] = p_collect_data2(DATASETS, 'train'); disp(['   Elapsed time ' num2str(toc) ' seconds.']);
% precompute TRAIN
if TRAIN.precomputed; p_precompute_features(TRAIN, LEARNERS); end

% collect the validation data set
tic; disp(['...collecting the ' num2str(DATASETS.VALIDATION_POS + DATASETS.VALIDATION_NEG) ' examples in the VALIDATION set.']);
[VALIDATION, DATASETS] = p_collect_data2(DATASETS, 'validation'); disp(['   Elapsed time ' num2str(toc) ' seconds.']);
VALIDATION.precomputed = 0;

%% ============================== train the cascade ==============================================================

CASCADE = p_cascade_init(DATASETS);       % initialize the CASCADE struct
i = 1;                                      % cascade stage index
Fi = 1; Di = 1;                             % cascade's current false positive / detection rates     

while (Fi > BOOST.targetF)                  % create new cascade stages until we meet target false positive rate
   
    disp(['================= TRAINING CASCADE STAGE i = ' num2str(i) ' =================']);
    ti = 1;                                 % weak learner index for cascade stage i
    restart_flag = 0;                       % =1 indicates problem with training, must retrain last weak learner
    CASCADE(i).di = 0;  CASCADE(i).fi = 0;  % di and fi are the false positive and detection rates for stage i
    
    % BOOSTING: add weak learners to stage i until dmin & fmax goals are met 
    while (CASCADE(i).fi > BOOST.goals(i).fmax) || (CASCADE(i).di < BOOST.goals(i).dmin)
        
        disp('   ----------------------------------------------------------------------------------------');
        disp(['   Training CASCADE Stage (i=' num2str(i) '): selecting weak learner (ti=' num2str(ti) ').']);
        
        % --------- train the next weak classifier ti for stage i --------- 
        CASCADE(i).CLASSIFIER = p_boost_step(TRAIN, LEARNERS, BOOST, ti, CASCADE, i);

        disp(' '); disp('       Class +1 Detection       Class -1 False Positives');     
        S1 = sprintf('       d*=%6.4g (minimum)\tf*=%6.4g (maximum)\t**STAGE %d GOALS**',BOOST.goals(i).dmin, BOOST.goals(i).fmax, i); disp(S1);
             
        % ------- adjust cascade threshold for stage i to meet dmin -------
        %  adjust the threshold for the current stage until we find one
        %  which gives a satifactory detection rate (this changes the false alarm rate)
        [CASCADE, Fi, Di]  = p_cascade_select_threshold(CASCADE, i, VALIDATION, BOOST.goals(i).dmin);

        
        % ...................... SANITY CHECK .............................
        % to make sure we're actually improving on the training data
        gt = [TRAIN(:).class]';  C = p_classify_cascade(CASCADE, TRAIN); [tpr fpr FPs TNs TPs] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist', 'TNlist', 'TPlist'); %disp(['Di=' num2str(tpr) ', Fi=' num2str(fpr) ', #FP = ' num2str(length(FPs)) '.  CASCADE applied to TRAIN set.'  ]);               
        S2 = sprintf('       Di=%5.4g (%d/%d)\tFi=%5.4g (%d/%d)\tCASCADE -> TRAIN SET', tpr, length(TPs), length(find(gt == 1)), fpr, length(FPs), length(find(gt == -1)) ); disp(S2);
        % .................................................................

        
        %...... HACK TO STOP IF REPEATING CLASSIFIER ........
        if (ti > 1) && strcmp(CASCADE(i).CLASSIFIER.learner_id{ti}, CASCADE(i).CLASSIFIER.learner_id{ti-1}) && (CASCADE(i).CLASSIFIER.threshold(ti) == CASCADE(i).CLASSIFIER.threshold(ti-1))
            disp(' REPEATED CLASSIFIER, ABORT!');
            beep; keyboard;
        end

        % write training results to the log file
        %logfile(EXPERIMENT.log_filenm, 'write', [i ti CASCADE(i).CLASSIFIER.feature_index(ti) Di Fi CASCADE(i).di CASCADE(i).fi tpr fpr length(FPs) CASCADE(i).CLASSIFIER.]);
        
        % save the cascade to a file in case something bad happens and we need to restart
        save(EXPERIMENT.cascade_filenm, 'CASCADE', 'EXPERIMENT', 'DATASETS', 'LEARNERS'); disp(' '); disp(['       ...saved a temporary copy of CASCADE to ' EXPERIMENT.cascade_filenm]);
    
        ti = ti + 1;        % proceed to the next weak learner
    end
    
    %% check to see if we have completed training
    if (Fi <= prod([BOOST.goals(:).fmax])) && (Di >= prod([BOOST.goals(:).dmin]))
        break;
    end

%     keyboard;
%     
%     if ~restart_flag
%         %% prepare training & validation data for the next stage of the cascade  
%         %  recollect negative examples for the training and validation set which 
%         %  include only FPs generated by the current cascade
%         disp('       ...updating the TRAIN set with negative examples which cause false positives');
%         TRAIN = p_collect_data(DATASETS, 'update', TRAIN, CASCADE, LEARNERS);
%         %TRAIN = ada_precompute(TRAIN, LEARNERS, WEAK, EXPERIMENT, 're');
% 
%         disp('       ...updating the VALIDATION set with negative examples which cause false positives');
%         VALIDATION = p_collect_data(DATASETS, 'update', VALIDATION, CASCADE, LEARNERS);
%         %VALIDATION = ada_precompute(VALIDATION, LEARNERS, WEAK, EXPERIMENT, 're');
%     else
%         % if we need to restart the stage we were just on
%         disp('       ...updating the TRAIN NEG set with a new set of false positives');
%         TRAIN = ada_collect_data(DATASETS, 'recollectFPs', TRAIN, CASCADE, LEARNERS);
%         TRAIN = ada_precompute(TRAIN, LEARNERS, WEAK, EXPERIMENT, 're');
%         disp('       ...updating the VALIDATION NEG set with a new set of false positives');
%         VALIDATION = ada_collect_data(DATASETS, 'recollectFPs', VALIDATION, CASCADE, LEARNERS);
%         VALIDATION = ada_precompute(VALIDATION, LEARNERS, WEAK, EXPERIMENT ,'re');
%     end
    
    i = i + 1;  % proceed to the next stage of the cascade
end


%% ============================== training complete, clean up & quit ==============================================================

disp('');
disp('==============================================================================');
disp(['Training complete.  CASCADE is stored in ' EXPERIMENT.cascade_filenm '.']);
disp('==============================================================================');
%clear TRAIN VALIDATION C gt NORM WEAK Di dmin Dlast Fi fmax Flast tpr fpr BOOST.targetF FPs i j ti log_filenm appname version author email IMSIZE TRAIN_POS TRAIN_NEG TEST_POS TEST_NEG cascade_filenm temppath temp_filenm datapath train1 train0 validation1 validation0 update0


