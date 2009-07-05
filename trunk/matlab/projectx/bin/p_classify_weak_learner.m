function h = p_classify_weak_learner(learner, polarity, threshold, SET, varargin)
%% TODO: write documenation
% returns a row vector, h

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

% get the feature responses to the (integral) images
f = double(p_RectangleFeature(SET.IntImages, {learner}));

% perform the weak classification to binary {0, 1}
h = ( polarity*f) < (polarity * threshold);

if (nargin == 5) && strcmp(varargin{1}, 'boolean')
    return;
else
    % convert classes to {-1, 1}
    h = double(h);
    h(h == 0) = -1;
end

