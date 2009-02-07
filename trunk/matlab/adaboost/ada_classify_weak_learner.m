function h = ada_classify_weak_learner(feature_index, weak_learner, SET)
%ADA_HAAR_TRCLASSIFY
%   Training classification - classify the entire training SET for a given
%   weak learner
%
%   h = ada_classify_weak_learner(feature_index, weak_learner, SET)
%
%
%
%
%


f = SET.responses.getCols(feature_index);
h = (weak_learner.polarity*ones(size(f)) .* f) <  ((weak_learner.polarity*ones(size(f))) .* (weak_learner.theta*ones(size(f))));

