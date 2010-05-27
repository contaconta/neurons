function E = HaarFeature(D, rects, cols)
%% AdaBoostClassify applies a boosted classifier to data  
%
%   AdaBoostClassify(rects, cols, thresh, pol, alpha, DATA)
%   TODO: write documentation.
%
%   See also 

%   Copyright © 2010 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.


% first, we can do some checks that the data is properly passed
rects = rects(:)';
cols = cols(:)';


E = HaarFeature_mex(D, rects, cols);