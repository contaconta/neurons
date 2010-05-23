function P = adaboost_classify(rects, pols, thresh, tpol, alpha, D)
%
%   
%

T = length(tpol); 

F = zeros(size(D,1), length(tpol), 'single');

for t = 1:T

    % 1. get the feature responses for each example in D
    F(:,t) = haar_feature(D, rects{t}, pols{t});

    % 2. perform the classification
    F(:,t) = tpol(t) * F(:,t) < tpol(t) * thresh(t);
    F(F == 0) = -1;     % set class 0 to class -1
    
end

P = (alpha(1:T) * F')';

P = single(P >= 0);  %.5 * sum(alpha(1:T)));
P(P == 0) = -1;          % set class 0 to class -1