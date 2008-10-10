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
%   See also ADA_COLLECT_DATA, ADA_DEFINE_CLASSIFIERS

%% set parameters and handle input arguments
PRE = varargin{1}; TRAIN = varargin{2}; WEAK = varargin{3}; T = varargin{4};

if nargin == 4
    % start new adaboost:  init training data weight vector.
    w = ones(1,length(TRAIN)) ./ length(TRAIN); 
    % Init a struct for the strong classifier, CLASSIFIER
    CLASSIFIER = ada_classifier_init(T, WEAK);
    tmin = 1;
else
    CLASSIFIER = varargin{5};  
    tmin = length(CLASSIFIER.feature_index) + 1;
    w = CLASSIFIER.w;
end

training_labels = [TRAIN(:).class];

%% train the strong classifier as a series of T weak classifiers
for t = tmin:T
    %% 1. Normalize the weights
    w = w ./sum(w);
    wstring = ['       t=' num2str(t) '/' num2str(T) ' optimized feature '];
    W = wristwatch('start', 'end', size(WEAK.descriptor,1), 'every', 10000, 'text', wstring);
    
    %% 2. find the optimal class separating theta and minerr for each feature
    for i = 1:(size(WEAK.descriptor,1))
        %[WEAK, PRE] = ada_find_haar_parameters(i, TRAIN, PRE, WEAK, w);
        [WEAK, PRE] = ada_find_haar_parameters2(i, training_labels, PRE, WEAK, w);
        W = wristwatch(W, 'update', i);
    end
    
    %% 3. Use the best WEAK classifier as the 't' hypothesis in the Strong CLASSIFIER
    [BEST_err, BEST_feature] = min(WEAK.minerr);
    
    CLASSIFIER.feature_index(t)         = BEST_feature; 
    CLASSIFIER.feature_descriptor(t,:)  = WEAK.descriptor(BEST_feature, :); 
    CLASSIFIER.fast(t,:)                = WEAK.fast(BEST_feature,:);
    CLASSIFIER.polarity(t)              = WEAK.polarity(BEST_feature); 
    CLASSIFIER.theta(t)                 = WEAK.theta(BEST_feature); 
    CLASSIFIER.alpha(t)                 = log( (1 - BEST_err) / BEST_err );
    beta = BEST_err/ (1 - BEST_err);      % beta is between [0, 1]
    disp(['...selected weak classifier ' num2str(BEST_feature) ' as t=' num2str(t)  '  [polarity = ' num2str(CLASSIFIER.polarity(t)) ' theta = ' num2str(CLASSIFIER.theta(t))  ']' ]);
    
    %% 4. Update the training weight vector according to misclassifications
    IIs = [TRAIN(:).II];                    % vectorize the integral images
    f = ada_fast_haar_response(CLASSIFIER.fast(t,:), IIs);
    h = (CLASSIFIER.polarity(t)*ones(size(f)) .* f) <  ((CLASSIFIER.polarity(t)*ones(size(f))) .* (CLASSIFIER.theta(t)*ones(size(f))));
    e = abs( h - [TRAIN(:).class] );
    w = w .* (beta * ones(size(w))).^(1 - e);
    CLASSIFIER.w = w;   
    clear IIs beta e f h    
end

