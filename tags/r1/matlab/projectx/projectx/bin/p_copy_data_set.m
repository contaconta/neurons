function SET = p_copy_data_set(SET, inds)
%P_GET_FEATURE_RESPONSES
%
%   TODO: documentation
%
%   Examples:
%   ----------------------
%
%   See also P_TRAIN, P_SETTINGS

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

flist = fields(SET);

N = length(SET.class);

for i=1:length(flist)
    if length(SET.(flist{i})) == N
        SET.(flist{i}) = SET.(flist{i})(inds);
    end
end
