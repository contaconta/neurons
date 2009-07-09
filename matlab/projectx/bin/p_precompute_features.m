function p_precompute_features(DATA_SET, LEARNERS)
%
%   TODO: WRITE DOC

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


for l = 1:length(LEARNERS.list)
    
    % switch LEARNERS.list(l)(1:2)
   
    % precompute the feature responses for each example for learner l
    %learner = LEARNERS.list{l};
    responses = p_RectangleFeature(SET.IntImages, LEARNERS.list(l));
    
    % store the responses as a row vector
    memClient.store(responses, 'row', l, 'HA');
end
