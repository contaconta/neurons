function CLASSIFIER = p_boost_step(TRAIN, LEARNERS, BOOST, ti, CASCADE, i)
%% P_BOOST_STEP adds a new weak learner using appropriate boosting method 
%
%   TODO: write documentation.
%
%   See also P_TRAIN, p_ADABOOST, p_REALBOOST, p_DPBOOST

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


if ti == 1
    % add the first weak learner to the classifier
    boost_step = BOOST.function_handle;
    CLASSIFIER = boost_step(TRAIN, LEARNERS, ti);
else
    % add subsequent weak learners to the classifier
    CLASSIFIER = CASCADE(i).CLASSIFIER;
    boost_step = BOOST.function_handle;
    CLASSIFIER = boost_step(TRAIN, LEARNERS, ti, CLASSIFIER);
end


