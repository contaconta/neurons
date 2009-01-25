function WEAK = ada_train_weak_learners(WEAK, TRAIN, w)
%
%
%
%
%
%
%





training_labels = [TRAIN.class];


%% loop through types of weak learners, learn optimal parameters for each to separate weighted training data
for l = 1:length(WEAK.learners)
    learner_type        = WEAK.learners{l}{1};
    field               = WEAK.learners{l}{2};
    map                 = WEAK.learners{l}{3};
    learning_function   = WEAK.learners{l}{4};
    num_learners        = length(map);
    
    
    
    if strcmp(learner_type, 'haar')
        wstring = '       optimized haar feature ';
        W = wristwatch('start', 'end', num_learners, 'every', 10000, 'text', wstring);
        for i = 1:num_learners
            [WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
                                   learning_function(map(i), training_labels, TRAIN, w);  % needs map(i) since grabbing f response from PRE
            W = wristwatch(W, 'update', i);
        end
    end
    
    if strcmp(learner_type, 'spedge')
        wstring = '       optimized spedge feature ';
        W = wristwatch('start', 'end', num_learners, 'every', 10000, 'text', wstring);
        for i = 1:num_learners
            [WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
                                   learning_function(map(i), training_labels, TRAIN, w);
            W = wristwatch(W, 'update', i);
        end
    end
    
end
