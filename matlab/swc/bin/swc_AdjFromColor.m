function W = swc_AdjFromColor(varargin)
%% SWC_CP forms connected components from G
%   CP = swc_CP(G, B)
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


if nargin == 4
    A = varargin{1};
    C = varargin{2};
    W = varargin{3};
    colors = varargin{4};
    colors = colors(:)';

    for c = colors 
        MASK = sparse([],[],[], size(W,1), size(W,1),0);
        members = find(C == c);
        W(members,:) = 0;
        W(:,members) = 0;
        MASK(members,members) = 1;
        Wc = A.*MASK;
        W(members,members) = Wc(members, members);
    end
    
elseif nargin == 2
    A = varargin{1};   % the full adjacency map G0
    C = varargin{2};   % a list of vertex colors
    
    W = sparse([],[],[], size(A,1), size(A,1),0);

    for c = 1:length(C)
        MASK = sparse([],[],[], size(W,1), size(W,1),0);
        members = find(C == c)';    % vertices belonging to color c
        MASK(members,members) = 1;  % only members rows & columns
        Wc = A.*MASK;               % a small fully connected graph for color c
        W = W + Wc;                 % add for each color to get W
    end

    
end