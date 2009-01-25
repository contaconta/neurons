function CLASSIFIER = ada_adaboost(varargin)
%% ADA_ADABOOST trains a strong classifier from weak classifiers & training data.
%
%   CLASSIFIER = ada_adaboost(PRE, TRAIN, WEAK, T) trains a strong
%   classifier CLASSIFIER from T hypotheses generated from weighted weak
%   classifiers WEAK on training examples from data stored in 
%   struct TRAIN.  T defines the number of hypotheses make up the strong 
%   classifier.  PRE is a bigmatrix containing precomputed feature responses 
%   to the training set.  You may resume training an existing classifier by
%   calling CLASSIFIER = ada_adaboost(PRE, TRAIN, WEAK, T, CLASSIFIER);.
%
%
%   Copyright © 2008 Kevin Smith
%   See also ADA_DEFINE_CLASSIFIERS, ADA_TRAIN_WEAK_LEARNERS


%% set parameters and handle input arguments
TRAIN = varargin{1}; WEAK = varargin{2}; T = varargin{3}; LEARNERS = varargin{4};

% either start or resume training, CLASSIFIER, w need to be passed or
% initialized. start_t is the index of the first weak learner.
if nargin == 4
    start_t = 1; w = ones(1,length(TRAIN.class)) ./ length(TRAIN.class); 
    CLASSIFIER = ada_classifier_init(T, WEAK);
else
   CLASSIFIER = varargin{5};  
   start_t = length(CLASSIFIER.feature_index) + 1;  w = CLASSIFIER.w;
end


%% train the strong classifier as a series of T weak classifiers
for t = start_t:T
    %% 1. Normalize the weights
    % normalize so each class has weight = 0.5
    w(TRAIN.class == 1) = .5 * (w(TRAIN.class==1) /sum(w(TRAIN.class==1)));
    w(TRAIN.class == 0) = .5 * (w(TRAIN.class==0) /sum(w(TRAIN.class==0)));
    
    %% 2. train weak learners for optimal class separation
    %WEAK = ada_train_weak_learners(WEAK, TRAIN, double(w));
    WEAK = ada_train_weak_learners(WEAK, TRAIN, w);
    
    %% 3. Use the best WEAK learner as the t-th CLASSIFIER hypothesis 
    [BEST_err, BEST_learner] = min(WEAK.error);
    
    % populate the selected classifier with needed information
    CLASSIFIER.feature_index(t) = BEST_learner; 
    alpha                       = log( (1 - BEST_err) / BEST_err );
    %CLASSIFIER.alpha(t)         = log( (1 - BEST_err) / BEST_err );
    beta                        = BEST_err/ (1 - BEST_err);      % beta is between [0, 1]
    weak_classifier             = WEAK.(WEAK.list{BEST_learner,1})(WEAK.list{BEST_learner,2});
    learner_ind                 = WEAK.list{BEST_learner,3};
    field                       = WEAK.learners{learner_ind}{1};
    classification_function     = WEAK.learners{learner_ind}{6};
    CLASSIFIER.learner_type{t}  = field;
    
    % add the weak learner to the classifier (if it is not the first of its
    % type it needs to be appended, otherwise added)
    if ~isfield(CLASSIFIER, field)
        CLASSIFIER.weak_learners{1} = weak_classifier;
        CLASSIFIER.weak_learners{1}.alpha = alpha;
        CLASSIFIER.weak_learners{1}.index = BEST_learner;
        CLASSIFIER.weak_learners{1}.type = field;
    else
        CLASSIFIER.weak_learners{t} = weak_classifier;
        CLASSIFIER.weak_learners{t}.alpha = alpha;
        CLASSIFIER.weak_learners{t}.index = BEST_learner;
        CLASSIFIER.weak_learners{t}.type = field;
    end
    
    % add the weak classification function to a list of classification functions
    if isempty(CLASSIFIER.functions)
        CLASSIFIER.functions    = {field classification_function};
    elseif ~ismember(field, CLASSIFIER.functions(:,1))
        CLASSIFIER.functions    = [CLASSIFIER.functions ; {field classification_function}];
    end

%     %%%%%%%%%%%%%%%%%%% DEBUG %%%%%%%%%%%%%%%%%%%%
%     tmpstring =[];
%     for i = 1:length(WEAK.learners)
%         [m1, v1] = min(WEAK.error(WEAK.learners{i}{3}));
%         s = ['best ' WEAK.learners{i}{1} ': ' num2str(v1) ' err=' num2str(m1) '   '];
%         tmpstring = [tmpstring s];
%     end
%     disp(tmpstring);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%% DEBUG %%%%%%%%%%%%%%%%%%%%
    for i = 1:length(WEAK.learners)
        [m1, v1] = min(WEAK.error(WEAK.learners{i}{3}));
        if m1 == min(WEAK.error); prestr = '     ✓ '; else prestr = '       '; end
        s = [prestr 'best ' WEAK.learners{i}{1} ' error: ' num2str(m1) ', feature index: ' num2str(v1)];
        disp(s);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %disp(['...selected learner ' num2str(BEST_learner) ' (' field ') as t=' num2str(t) ]);
    
    
    %% 4. Update the training weight vector according to misclassifications
    % get selected weak learner's classification results for the TRAIN set
    %h = classification_function(weak_classifier, TRAIN, [0 0], CLASSIFIER.IMSIZE);
    h = classification_function(CLASSIFIER.feature_index(t), weak_classifier, TRAIN)';
    
    % reweight misclassified examples to be more important (& store)
    e = abs( h - TRAIN(:).class);
    w = w .* (beta * ones(size(w))).^(1 - e);
    CLASSIFIER.w = w;   
    
    % clean up
    clear IIs beta e f h    
end






