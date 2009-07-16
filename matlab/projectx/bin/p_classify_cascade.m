function C = p_classify_cascade(CLASSIFIER, SET, varargin)
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


% if a single stage is given, make it appear like a cascade
if strcmp(CLASSIFIER(1).type, 'SINGLE');
    CASCADE.CLASSIFIER = CLASSIFIER;
else
    CASCADE = CLASSIFIER;
end

% initialize the classification vector to be all positive classes
C =  ones(size(SET.class));

% loop through each stage of the cascade (even if it is only 1 stage)
for s = 1:length(CASCADE)
    
    % make a list of examples which are still positive
    positive_list = find(boolean(C));
    
    % handle the case where there are no positive examples left
    if isempty(positive_list); C = zeros(size(C)); disp('   no positives left!'); break; end

    % make a copy of the data set with only examples that passed to stage s
    STAGE_S = p_copy_data_set(SET, positive_list);
    h = zeros(length(positive_list), length(CASCADE(s).CLASSIFIER.learner));
    
    % weak hypotheses for stage s
    for ti = 1:length(CASCADE(s).CLASSIFIER.learner)
        h(:,ti) = p_classify_weak_learner(CASCADE(s).CLASSIFIER.learner{ti}, CASCADE(s).CLASSIFIER.polarity(ti), CASCADE(s).CLASSIFIER.threshold(ti), STAGE_S, 'boolean');
    end
    
   % keyboard;
    
    %% compute the strong classification
    % the strong classification is computed by the weighted sum of the weak
    % classification results.  if this is > .5 * sum(alpha), it is class 1,
    % otherwise it is class 0.  varying 'threshold' (default = 1) adjusts 
    % the sensitivity of the strong classifier
    alpha = repmat(CASCADE(s).CLASSIFIER.alpha, length(positive_list), 1);
    ha = h .* alpha;
    asum = sum(alpha,2);
    
    if nargin > 2; threshold = varargin{1}; else threshold = CASCADE(s).threshold; end
    
    % Cstage is a column vector containing strong classification results
    % for stage s.  each row represents the classification of a training
    % example
    Cstage = zeros(size(C));
    Cstage(positive_list) = sum(ha,2) > (.5 * asum * threshold);

    % an AND operation updates newly found rejections
    C = C & Cstage;
    
end