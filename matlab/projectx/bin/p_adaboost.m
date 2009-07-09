function CLASSIFIER = p_adaboost(varargin)
%% ADA_ADABOOST trains a strong classifier from weak classifiers & training data.
%
%   CLASSIFIER = p_adaboost(TRAIN, WEAK, ti, LEARNERS) trains a strong
%   classifier CLASSIFIER from T hypotheses generated from weighted weak
%   classifiers WEAK on training examples from data stored in 
%   struct TRAIN.  ti defines the number of hypotheses make up the strong 
%   classifier.  PRE is a bigmatrix containing precomputed feature responses 
%   to the training set.  You may resume training an existing classifier by
%   calling CLASSIFIER = ada_adaboost(PRE, TRAIN, WEAK, ti, CLASSIFIER);.

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


%% set parameters and handle input arguments
TRAIN = varargin{1}; LEARNERS = varargin{2}; ti = varargin{3}; restart_flag = 0;

% either start or resume training, CLASSIFIER, w need to be passed or
% initialized. start_t is the index of the first weak learner.
if nargin == 3
    w = ones(1,length(TRAIN.class)) ./ length(TRAIN.class); 
    CLASSIFIER = [];
else
   CLASSIFIER = varargin{4};  
   w = CLASSIFIER.w;
end

% ---- when a new stage is reached, reset the example weights! ----
if ti == 1
    disp('   NORMALIZED THE WEIGHTS!');
    w = ones(1,length(TRAIN.class)) ./ length(TRAIN.class); 
end

%% 1. Normalize the weights
% normalize so each class has weight = 0.5
w(TRAIN.class == 1) = .5 * (w(TRAIN.class==1) /sum(w(TRAIN.class==1)));
w(TRAIN.class == -1) = .5 * (w(TRAIN.class==-1) /sum(w(TRAIN.class==-1)));

%% 2. select weak learner parameters for optimal class separation
W = wristwatch('start', 'end', length(LEARNERS.list), 'every', 10000);
for l = 1:length(LEARNERS.list)
    W = wristwatch(W, 'update', l, 'text', '       optimized feature ');
    [LEARNERS.error(l), LEARNERS.threshold(l), LEARNERS.polarity(l)] = p_select_weak_parameters(LEARNERS.list{l}, TRAIN, w);    
end


%% 3. Use the best WEAK learner as the ti-th CLASSIFIER hypothesis 
[MINerr, BESTlearner] = min(LEARNERS.error);


%% !!!!!!!!!! HERE WE MIGHT NEED TO ADD CODE TO DEAL WITH REPEATEDLY SELECTED CLASSIFIERS


% populate the selected classifier with needed information
%CLASSIFIER.feature_index(ti)    = BESTlearner; 
alpha                           = log( (1 - MINerr) / MINerr );
beta                            = MINerr/ (1 - MINerr);      % beta is between [0, 1]

CLASSIFIER.learner{ti}          = LEARNERS.list{BESTlearner};
CLASSIFIER.polarity(ti)         = LEARNERS.polarity(BESTlearner);
CLASSIFIER.threshold(ti)        = LEARNERS.threshold(BESTlearner);
CLASSIFIER.alpha(ti)            = alpha;


% append the weak learner to the classifier's list of weak learners
%weak_classifier                 = LEARNERS.list{BESTlearner};
%CLASSIFIER.weak_learners{ti} = weak_classifier;
%CLASSIFIER.weak_learners{ti}.alpha = alpha;
%CLASSIFIER.weak_learners{ti}.index = BESTlearner;



%.................. DISPLAY .....................
learner_string = LEARNERS.list{BESTlearner};
switch learner_string(1:2)
    case 'HA'
        feature_type = 'Haar-like';
    case 'RA' 
        feature_type = 'Rays';
end
s = ['       ✓ SELECTED ' feature_type ' learner, error: ' num2str(MINerr) ', polarity: ' num2str(CLASSIFIER.polarity(ti)) ', threshold: ' num2str(CLASSIFIER.threshold(ti))  ]; disp(s);
s = ['                  id: ' learner_string ]; disp(s);
%................................................




%% 4. Update the training weight vector according to misclassifications
% get selected weak learner's classification results for the TRAIN set
%h = ada_classify_weak_learner(BESTlearner, weak_classifier, TRAIN)';
%h = dummy_classify_set(CLASSIFIER.feature{ti}, TRAIN);
h = p_classify_weak_learner(CLASSIFIER.learner{ti}, CLASSIFIER.polarity(ti), CLASSIFIER.threshold(ti), TRAIN)';


% reweight misclassified examples to be more important (& store)
e = h ~= TRAIN.class;           %abs( h - TRAIN(:).class);
w = w .* (beta * ones(size(w))).^(1 - e);
CLASSIFIER.w = w;   








