function P = test_probmap(CLASSIFIER, data)
%% TODO: write documenation

%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%               Aurelien Lucchi     http://cvlab.epfl.ch/~lucchi/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.



h = zeros(1, length(CLASSIFIER.learner));
   
% weak hypotheses for each weak learner
for ti = 1:length(CLASSIFIER.learner)
    h(:,ti) = classify_weak(CLASSIFIER.learner{ti}, CLASSIFIER.polarity(ti), CLASSIFIER.threshold(ti), data, 'boolean');
end

%% compute the strong classification
% the strong classification is computed by the weighted sum of the weak
% classification results.  if this is > .5 * sum(alpha), it is class 1,
% otherwise it is class 0.  varying 'threshold' (default = 1) adjusts 
% the sensitivity of the strong classifier
alpha = repmat(CLASSIFIER.alpha, 1, 1);
ha = h .* alpha;
%asum = sum(alpha,2);

P = sum(ha,2);

% % Cstage is a column vector containing strong classification results
% % for stage s.  each row represents the classification of a training
% % example
% Cstage = zeros(size(C));
% Cstage(positive_list) = sum(ha,2) > (.5 * asum * threshold);
% 
% % an AND operation updates newly found rejections
% C = C & Cstage;

    
    
    
%% function to perform weak classfication using a weak learner 
function h = classify_weak(learner, polarity, threshold, data, varargin)

switch learner(1:2)
  case 'HA'
        f = mexRectangleFeature({data}, {learner});
  case 'IT'
	%display 'IT'
	%size(data)
	%learner
        f = mexIntensityFeature({data}, {learner});    
  otherwise
    error('could not find appropriate function for learner');
end

% perform the weak classification to binary {0, 1}
h = ( polarity*f) < (polarity * threshold);

if (nargin == 5) && strcmp(varargin{1}, 'boolean')
    return;
else
    % convert classes to {-1, 1}
    h = double(h);
    h(h == 0) = -1;
end

        
