function gplotregion(W, xy, C, cs, color, varargin)
%% NEIGHBORS2 finds neighbor elements in a 2D array
%   
%   TODO: write documentation

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

cs = cs(:)';

if nargin > 5
    linespc = varargin{1};
else
    linespc = 'o-';
end

for c = cs 
    MASK = sparse([],[],[], size(W,1), size(W,1),0);
    members = find(C == c)';
    MASK(members,members) = 1;
    A = W.*MASK;
    gplot2(A, [xy(:,2) xy(:,1)], linespc, 'Color', color);
end
