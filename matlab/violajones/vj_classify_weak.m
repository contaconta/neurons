function [h, PRE] = vj_classify_weak(varargin)
%VJ_CLASSIFY_WEAK returns the classification result of a weak classifier 
%
%   h = vj_classify_weak(descriptor, polarity, theta, II) given the feature
%   descriptor, polarity, the threshold 'theta' of a weak descriptor,  and an
%   integral image II of the patch to be classified, returns the image 
%   classification 'h' (class h=1 or class h=0).  I and II are assumed to be
%   resized to the IMSIZED used to define the weak classifiers.  Many weak 
%   classification hypotheses are used to build a strong classfier in adaboost.
%
%   Example: Calling vj_classify_weak normally.
%   -----------------------------------------------
%   h = vj_classify_weak(descriptor, polarity, theta, II);
%
%   Example: Calling vj_classify_weak if you have precomputed feature responses
%   -----------------------------------------------------------------------
%   h = vj_classify_weak(descriptor, polarity, theta, ...
%                        'precomputed', feature_ind, training_ind, PRE);
%
%   Copyright © 2008 Kevin Smith
%   See also VJ_TRAIN, VJ_ADABOOST, VJ_DEFINE_CLASSIFIERS, VJ_STRONG_CLASSIFY

descriptor = varargin{1};
polarity = varargin{2};
theta = varargin{3};

% if we've precomputed feature responses, we can save some computations
if strcmp(varargin{4}, 'precomputed')
    f_ind = varargin{5};
    train_ind = varargin{6};
    PRE = varargin{7};
    [f, PRE] = bigmatrix_get_row(PRE, f_ind, [train_ind], 'v');
else
    II = varargin{4};
    f = vj_haar_response(descriptor, II);   % compute the feature response
end



if (polarity * f) < (polarity * theta)
    h = 1;                  % class 1 if the feature response < threshold
else
    h = 0;                  % class 0 if the feature response >= threshold
end
    