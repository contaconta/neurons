function CASCADE = ada_cut_cascade(CASCADE, nlearners)
%
%   CASCADE = ada_cut_cascade(CASCADE, nlearners)
%
%
%
%
%


learner_count = 0;

for s = 1:length(CASCADE)
    for i = 1:length(CASCADE(s).CLASSIFIER.weak_learners)

        learner_count = learner_count + 1;
        
        if learner_count >= nlearners
            disp(['  ... cutting the cascade to ' num2str(nlearners) ' learners'])
            break;
        end
        
    end
    
    if learner_count >= nlearners
        %disp(['  ... cutting the cascade to ' num2str(nlearners) ' learners'])
        break;
    end
end


CASCADE = CASCADE(1:s);
CASCADE(s).CLASSIFIER.feature_index = CASCADE(s).CLASSIFIER.feature_index(1:i);
CASCADE(s).CLASSIFIER.polarity = CASCADE(s).CLASSIFIER.polarity(1:i);
CASCADE(s).CLASSIFIER.theta = CASCADE(s).CLASSIFIER.theta(1:i);
CASCADE(s).CLASSIFIER.alpha = CASCADE(s).CLASSIFIER.alpha(1:i);

% refill the weak_learners
learners = CASCADE(s).CLASSIFIER.weak_learners;
CASCADE(s).CLASSIFIER.weak_learners = {};

for k = 1:i
    CASCADE(s).CLASSIFIER.weak_learners{k} = learners{k};
end

