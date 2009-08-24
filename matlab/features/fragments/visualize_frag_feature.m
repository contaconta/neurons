function visualize_frag_feature(IMSIZE, learner_string, learner_data)
%VISUALIZE_FRAG_FEATURE
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

A = uint8(zeros(IMSIZE));

% extract the BB from the learner string
m = regexp(learner_string, 'FR_(\d*)?_p_(\d*)?_img_(\d*)?_bb_(\d*)_(\d*)_(\d*)_(\d*)', 'tokens');
BB = [str2double(m{1}{4})  str2double(m{1}{5}) str2double(m{1}{6}) str2double(m{1}{7})];


% scale the fragment to the appropriate size, if necessary
if ~isequal(size(learner_data), [BB(4)+1 BB(3)+1])
    fragment = imresize(learner_data, [BB(4)+1 BB(3)+1]);
else
    fragment = learner_data;
end

if (BB(2)+BB(4) > size(A,1)) || (BB(1)+BB(3)) > size(A,2)
    error('FragFeature Error: bounding box is out of range of mask');
end

A(BB(2):BB(2)+BB(4), BB(1):BB(1)+BB(3)) = fragment;



pause(0.01);
figure(1);
set(gca, 'Position', [0 0 1 1]);
imshow(A);
refresh;

