function responses = fragFeature(Images, learner_ids, learner_data)
%FRAGFEATURE
%   See also FRAG_FEATURE

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


% pre-allocate the responses array (row vector)
%responses = int32(zeros([1 length(Images)]));     % CAST to INT32
responses = zeros([1 length(Images)]);

% loop through the cell of Images
for i=1:length(Images)
   
    % parse the learner_id to get parameters for the feature
    m = regexp(learner_ids{1}, 'FR_(\d*)?_p_(\d*)?_img_(\d*)?_bb_(\d*)_(\d*)_(\d*)_(\d*)', 'tokens');
    BB = [str2double(m{1}{4})  str2double(m{1}{5}) str2double(m{1}{6}) str2double(m{1}{7})];
    p = str2double(m{1}{2});
    %id = a(1);
    %img = a(2);
    
    % compute the feature response and store it in responses
    responses(i) = frag_feature(Images{i}, learner_data{1}, BB, p); 
   
    %disp([ 'Image i=' num2str(i) ' learner_id=' learner_ids{1} ]);
    
end