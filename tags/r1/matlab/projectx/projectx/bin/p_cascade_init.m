function CASCADE =p_cascade_init(DATASETS)
%% P_CASCADE_INIT initializes a cascade classifier structure 
%
%   CASCADE = p_cascade_init(DATASETS) creates a new structure to represent
%   the boosted cascaded classifier.  Each stage of the cascade can be
%   accessed by CASCADE(i).CLASSIFIER where i is the index of the stage.
%
%   See also P_TRAIN

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

CASCADE.CLASSIFIER = [];            % the first classifier is left empty
CASCADE.fi = 1;                     % this classifiers false alarm rate
CASCADE.di = 1;                     % this classifiers detection rate
CASCADE.threshold = 2  ;            % the starting sliding adaboost threshold for this classifier
CASCADE.type = 'CASCADE';           % specify if this is a cascade or single classifier
CASCADE.dataset = DATASETS.labelme_pos_query;% specify the data set the cascade was trained on
