function [C, h] = ada_classify_strong(CLASSIFIER, II, offset, threshold)
%ADA_CLASSIFY_STRONG returns the classification result of a boosted classifier
%
%   [C, h] = ada_classify_weak(CLASSIFIER, II, offset, threshold) given a 
%   boosted CLASSIFIER and a query integral image II, returns the boosted 
%   classification result (C = 1 for class 1 or C = 0 for class 0). II is
%   assumed to be the IMSIZE used to define the weak classifiers.  Many 
%   weak classification hypotheses are used to build a strong classfier 
%   in adaboost.  Offset translates the descriptor [x y] along the image.
%   Threshold adjusts the sensitivity of the classifier ( > 1 is more 
%   selective, < 1 is more permissive).
%  
%   Example: using ada_classify_strong to classify a query image.
%   -----------------------------------------------
%   C = ada_classify_strong(CLASSIFIER, II, [0 0])
%
%   Copyright © 2008 Kevin Smith
%   See also ADA_ADABOOST

x = offset(1);
y = offset(2);

if nargin < 3
    x = 0;  y = 0;
    threshold = 1;
elseif nargin < 4
    threshold = 1;
end


if size(II,2) ~= 1
    II = II(y+1:y+CLASSIFIER.IMSIZE(2), x+1:x+CLASSIFIER.IMSIZE(1));
    II = II(:);
end

f = ada_fast_haar_response(CLASSIFIER.fast, II);
h = (CLASSIFIER.polarity .* f) < (CLASSIFIER.polarity .* CLASSIFIER.theta);


% the final strong classifier is computed by the weighted sum of the weak
% classification results.  if this is > .5 * sum(alpha), it is class 1,
% otherwise it is class 0.  'threshold' (default = 1) allows us to adjust 
% the sensitivity of the classifier

if sum(CLASSIFIER.alpha.*h) >= (.5*threshold) * sum(CLASSIFIER.alpha) 
    C = 1;   %disp(['... boosted classification result C = ' num2str( sum(CLASSIFIER.alpha.*h)/sum(CLASSIFIER.alpha)) ' > .5  --->  Class 1']); 
else
    C = 0;   %disp(['... boosted classification result C = ' num2str( sum(CLASSIFIER.alpha.*h)/sum(CLASSIFIER.alpha)) ' < .5  --->  Class 0']);
end
