function learners = ada_learner_list(CASCADE)
%len = ada_cascade_length(CASCADE)
%
%   given a cascade, returns the total number of weak learners.
%
%

len = 0;
learners = [];

for i=1:length(CASCADE)
    for ti =1:length(CASCADE(i).CLASSIFIER.weak_learners)
        len = len + 1;
        learners{len} = CASCADE(i).CLASSIFIER.weak_learners{ti};
    end
end