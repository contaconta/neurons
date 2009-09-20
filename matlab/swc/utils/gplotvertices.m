function gplotvertices(W, xy, LABELS, Iraw, V, color, varargin)
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

V0 = V0(:)';
nL = length(V0);

if nargin > 6
    linespc = varargin{1};
else
    linespc = 'o-';
end

% plot the labeled graph
%imshow(Iraw); axis image off; set(gca, 'Position', [0 0 1 1]); hold on; 

for l = V0
    MASK = sparse([],[],[], size(W,1), size(W,1),0);
    members = find(LABELS == l)';
    MASK(members,members) = 1;
    A = W.*MASK;
    gplot2(A, [xy(:,2) xy(:,1)], linespc, 'Color', color);
end