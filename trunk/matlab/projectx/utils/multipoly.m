function [node, cnect] = multipoly(xcell, ycell)
%% MULTIPOLY converts LabelMe multiple polygons into format for inpoly
%
%
%   See also INPOLY

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


node = [];
cnect = [];

for i = 1:length(xcell)
    
    n = size(node,1)+1;
    N = length(xcell{i});
    
    node(n:n+N-1,1) = xcell{i};
    node(n:n+N-1,2) = ycell{i};
    cnect(n:n+N-1,:) = [(n:n+N-2)' (n+1:n+N-1)'; n+N-1 n];
    
end
