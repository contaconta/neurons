function CLASSIFIER = ada_adaboost(varargin)
%% ADA_ADABOOST trains a strong classifier from weak classifiers & training data.
%
%   CLASSIFIER = ada_adaboost(TRAIN, WEAK, ti, LEARNERS) trains a strong
%   classifier CLASSIFIER from T hypotheses generated from weighted weak
%   classifiers WEAK on training examples from data stored in 
%   struct TRAIN.  ti defines the number of hypotheses make up the strong 
%   classifier.  PRE is a bigmatrix containing precomputed feature responses 
%   to the training set.  You may resume training an existing classifier by
%   calling CLASSIFIER = ada_adaboost(PRE, TRAIN, WEAK, ti, CLASSIFIER);.
%
%
%   Copyright © 2008 Kevin Smith
%   See also ADA_DEFINE_CLASSIFIERS, ADA_TRAIN_WEAK_LEARNERS


%% set parameters and handle input arguments
TRAIN = varargin{1}; WEAK = varargin{2}; ti = varargin{3}; LEARNERS = varargin{4};

% either start or resume training, CLASSIFIER, w need to be passed or
% initialized. start_t is the index of the first weak learner.
if nargin == 4
    start_t = 1; w = ones(1,length(TRAIN.class)) ./ length(TRAIN.class); 
    CLASSIFIER = ada_classifier_init(ti, WEAK);
else
   CLASSIFIER = varargin{5};  
   start_t = length(CLASSIFIER.feature_index) + 1;  w = CLASSIFIER.w;
end


%% train the strong classifier as a series of ti weak classifiers
for t = start_t:ti
    %% 1. Normalize the weights
    % normalize so each class has weight = 0.5
    w(TRAIN.class == 1) = .5 * (w(TRAIN.class==1) /sum(w(TRAIN.class==1)));
    w(TRAIN.class == 0) = .5 * (w(TRAIN.class==0) /sum(w(TRAIN.class==0)));
    
    %% 2. train weak learners for optimal class separation
    WEAK = ada_train_weak_learners(WEAK, TRAIN, w);
    
%     %================= debug ===============
%     CLASSIFIER.wlog(t,:) = w; 
%     wlog = CLASSIFIER.wlog;
%     save([pwd '/tmp/WLOG' num2str(t) '.mat'], 'wlog');
%     
%     if t >= 46
%         disp('about to start repeating!');  keyboard;
%     end
%     %=======================================
    
    
    %% 3. Use the best WEAK learner as the t-th CLASSIFIER hypothesis 
    [BEST_err, BEST_learner] = min(WEAK.error);
    
    %======== HACK to avoid repeatedly selecting same feture ==============
    if (t > 1) && (BEST_learner == CLASSIFIER.feature_index(t-1))

        disp(' !!!! REPEATED CLASSIFIER!!!! ');
        % if we have a repeated classifier, set the weight of the leading
        % classifier to 0.
        maxinds = find(w == max(w));
        w(w == max(w)) = 0;
        disp(['set leading weights for examples [' num2str(maxinds) '] to 0.']);
        fid = fopen('BADEXAMPLES.txt', 'a', 'n');
        cstring = [TRAIN.database ' bad example: ' num2str(maxinds) sprintf('\n')];
        fwrite(fid, cstring);
        fclose(fid);  
    end
    %======================================================================
    
    
    % populate the selected classifier with needed information
    CLASSIFIER.feature_index(t) = BEST_learner; 
    alpha                       = log( (1 - BEST_err) / BEST_err );
    beta                        = BEST_err/ (1 - BEST_err);      % beta is between [0, 1]
    weak_classifier             = WEAK.learners{BEST_learner};
    CLASSIFIER.polarity(t)      = weak_classifier.polarity;
    CLASSIFIER.theta(t)         = weak_classifier.theta;
    CLASSIFIER.alpha(t)         = alpha;
   
    
    % append the weak learner to the classifier's list of weak learners
    CLASSIFIER.weak_learners{t} = weak_classifier;
    CLASSIFIER.weak_learners{t}.alpha = alpha;
    CLASSIFIER.weak_learners{t}.index = BEST_learner;
        

    %%%%%%%%%%%%%%%%%%% DISPLAY %%%%%%%%%%%%%%%%%%%
    for i = 1:length(LEARNERS)
        type = LEARNERS(i).feature_type;  [m1, v1] = min(WEAK.error( WEAK.lists.(type) )); ind = WEAK.lists.(type)(v1);
        if m1 == min(WEAK.error); prestr = '     ✓ '; else prestr = '       '; end
        s = [prestr 'best ' type ' error: ' num2str(m1) ', feature index: ' num2str(ind)]; disp(s);
    end
    s = ['       SELECTED ' WEAK.learners{BEST_learner}.type ' error: ' num2str(BEST_err) ', feature index: ' num2str(BEST_learner) ', polarity: ' num2str(CLASSIFIER.polarity(t)) ', theta: ' num2str(CLASSIFIER.theta(t))]; disp(s);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   

    
    %% 4. Update the training weight vector according to misclassifications
    % get selected weak learner's classification results for the TRAIN set
    h = ada_classify_weak_learner(BEST_learner, weak_classifier, TRAIN)';
    
    % reweight misclassified examples to be more important (& store)
    e = abs( h - TRAIN(:).class);
    w = w .* (beta * ones(size(w))).^(1 - e);
    CLASSIFIER.w = w;   
    
    % clean up
    clear IIs beta e f h    
end

