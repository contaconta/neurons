function len = ada_cascade_length(CASCADE)
%len = ada_cascade_length(CASCADE)
%
%   given a cascade, returns the total number of weak learners.
%
%

len = 0;

for i=1:length(CASCADE)
    len = len + length(CASCADE(i).CLASSIFIER.weak_learners);
end

