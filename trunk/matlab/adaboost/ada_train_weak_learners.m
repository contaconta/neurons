function [WEAK, PRE] = ada_train_weak_learners(WEAK, TRAIN, PRE, w)
%
%
%
%
%
%
%


% %% 1. some initialization for the wristwatch
% t=1;T=1;wstring = ['       t=' num2str(t) '/' num2str(T) ' optimized feature '];
% W = wristwatch('start', 'end', size(WEAK.error,1), 'every', 10000, 'text', wstring);
% 
% 

%% 2. determine the optimal class separating parameters and minimum error for each feature
training_labels = [TRAIN(:).class];

for l = 1:length(WEAK.learners)
    learner_type        = WEAK.learners{l}{1};
    field               = WEAK.learners{l}{2};
    map                 = WEAK.learners{l}{3};
    learning_function   = WEAK.learners{l}{4};
    num_learners        = length(map);
    
    
    
    if strcmp(learner_type, 'haar')
        t=1;T=1;wstring = ['       t=' num2str(t) '/' num2str(T) ' optimized haar feature '];
        W = wristwatch('start', 'end', num_learners, 'every', 10000, 'text', wstring);
        for i = 1:num_learners
            [PRE, WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
                                   learning_function(map(i), training_labels, PRE, WEAK, w);  % needs map(i) since grabbing f response from PRE
            W = wristwatch(W, 'update', map(i));
        end
    end
    
    if strcmp(learner_type, 'spedge')
        PRE = [TRAIN(:).sp];
        t=1;T=1;wstring = ['       t=' num2str(t) '/' num2str(T) ' optimized spedge feature '];
        W = wristwatch('start', 'end', num_learners, 'every', 1000, 'text', wstring);
        for i = 1:num_learners
            [WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
                                   learning_function(i, field, TRAIN, PRE, WEAK, w);
            W = wristwatch(W, 'update', i);
        end
    end
    
end
