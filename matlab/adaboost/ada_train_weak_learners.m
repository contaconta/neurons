function WEAK = ada_train_weak_learners(WEAK, TRAIN, w)
%ADA_TRAIN_WEAK_LEARNERS
%
%   WEAK = ada_train_weak_learners(WEAK, TRAIN, w) takes inputs weak learner
%   struct WEAk and training data TRAIN.  For each defined weak learner, it
%   applies the appropriate optimization function so that the weak learner
%   will compute its optimal parameters (theta, polarity).
%
%





training_labels = [TRAIN.class];


W = wristwatch('start', 'end', length(WEAK.learners), 'every', 10000);

%% loop through types of weak learners, learn optimal parameters for each to separate weighted training data
for i = 1:length(WEAK.learners)
    learner_type        = WEAK.learners{i}.type;
    
    switch learner_type
        case 'intmean'
            wstring = '       optimized intmean feature ';
            W = wristwatch(W, 'update', i, 'text', wstring);
            [WEAK.error(i), WEAK.learners{i}.theta, WEAK.learners{i}.polarity] = ada_weak_learn(i, training_labels, TRAIN, w);
          
        case 'intvar'
            wstring = '       optimized intvar feature ';
            W = wristwatch(W, 'update', i, 'text', wstring);
            [WEAK.error(i), WEAK.learners{i}.theta, WEAK.learners{i}.polarity] = ada_weak_learn(i, training_labels, TRAIN, w);
               
        case 'haar'
            wstring = '       optimized haar feature ';
            W = wristwatch(W, 'update', i, 'text', wstring);
            [WEAK.error(i), WEAK.learners{i}.theta, WEAK.learners{i}.polarity] = ada_weak_learn(i, training_labels, TRAIN, w);
            
        case 'spedge'
            wstring = '       optimized spedge feature ';
            W = wristwatch(W, 'update', i, 'text', wstring);
            [WEAK.error(i), WEAK.learners{i}.theta, WEAK.learners{i}.polarity] = ada_weak_learn(i, training_labels, TRAIN, w);
            
        case 'spdiff'
            wstring = '       optimized spdiff feature ';
            W = wristwatch(W, 'update', i, 'text', wstring);
            [WEAK.error(i), WEAK.learners{i}.theta, WEAK.learners{i}.polarity] = ada_weak_learn(i, training_labels, TRAIN, w);
        case 'hog'
            wstring = '       optimized hog feature ';
            W = wristwatch(W, 'update', i, 'text', wstring);
            [WEAK.error(i), WEAK.learners{i}.theta, WEAK.learners{i}.polarity] = ada_weak_learn(i, training_labels, TRAIN, w);
    
    
    end
end






% learner_type        = WEAK.learners{l}{1};
%     field               = WEAK.learners{l}{2};
%     map                 = WEAK.learners{l}{3};
%     learning_function   = WEAK.learners{l}{4};
%     num_learners        = length(map);
%     

% if strcmp(learner_type, 'haar')
%         wstring = '       optimized haar feature ';
%         W = wristwatch('start', 'end', num_learners, 'every', 10000, 'text', wstring);
%         for i = 1:num_learners
%             [WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
%                                    learning_function(map(i), training_labels, TRAIN, w);  % needs map(i) since grabbing f response from PRE
%             W = wristwatch(W, 'update', i);
%         end
%     end
%     
%     if strcmp(learner_type, 'spedge')
%         wstring = '       optimized spedge feature ';
%         W = wristwatch('start', 'end', num_learners, 'every', 10000, 'text', wstring);
%         for i = 1:num_learners
%             [WEAK.error(map(i)), WEAK.(field)(i).theta, WEAK.(field)(i).polarity] = ...
%                                    learning_function(map(i), training_labels, TRAIN, w);
%             W = wristwatch(W, 'update', i);
%         end
%     end


