function h = ada_haar_trclassify1(feature_index, weak_learner, SET)
%ADA_HAAR_TRCLASSIFY
%   Training classification - classify the entire training SET for a given
%   weak learner
%
%   h = ada_haar_trclassify1(weak_learner, SET, IMSIZE)
%
%
%
%
%


f = SET.responses.getCols(feature_index);
h = (weak_learner.polarity*ones(size(f)) .* f) <  ((weak_learner.polarity*ones(size(f))) .* (weak_learner.theta*ones(size(f))));

