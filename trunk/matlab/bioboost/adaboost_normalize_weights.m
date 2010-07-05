function W = adaboost_normalize_weights(W)
%   Normalize the weights so that each class +1 / -1 has 50% weight
%
%

W = W/sum(W);

%W(L == 1) = .5 * (W(L == 1) / sum(W(L == 1)));
%W(L == -1) = .5 * (W(L == -1) / sum(W(L == -1)));