function [WEAK, PRE] = ada_train_weak_learners(WEAK, TRAIN, PRE, w)
%
%
%
%
%
%
%


%% 1. some initialization for the wristwatch
t=1;T=1;wstring = ['       t=' num2str(t) '/' num2str(T) ' optimized feature '];
W = wristwatch('start', 'end', size(WEAK.error,1), 'every', 10000, 'text', wstring);



%% 2. optimal class separating theta and minerr for each feature

training_labels = [TRAIN(:).class];

for l = 1:length(WEAK.learners)
    learner_type        = WEAK.learners{l}{1};
    field               = WEAK.learners{l}{2};
    map                 = WEAK.learners{l}{3};
    num_learners        = length(map);
    learning_function   = WEAK.learners{l}{4};
    
    
    if strcmp(learner_type, 'haar')
        for i = 1:num_learners
            [PRE, WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
                                   learning_function(map(i), training_labels, PRE, WEAK, w);
            W = wristwatch(W, 'update', map(i));
        end
    end
    
    if strcmp(learner_type, 'spedge')
        for i = 1:num_learners
            [WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
                                   learning_function(map(i), TRAIN, WEAK, w);
            W = wristwatch(W, 'update', map(i));
        end
    end
    
end
