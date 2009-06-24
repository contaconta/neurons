function LEARNERS = p_enumerate_learners(LEARNERS, DATASETS)
%% P_ENUMERATE_LEARNERS is a temporary function until c++ implementation
%
%   LEARNERS = p_enumerate_learners(LEARNERS) reads the learner types from 
%   LEARNERS and enumerates individual weak learners, defining the
%   parameters of each weak learner.
%
%   See also P_SETTINGS, P_TRAIN

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


count = 1;

for l = 1:length(LEARNERS.types)
    
    str = LEARNERS.types{l};
    
    
    for i = 1:10000
        LEARNERS.list{count} = [str(1:3) number_into_string(i, 100000)];  
        count = count + 1;
    end
end
