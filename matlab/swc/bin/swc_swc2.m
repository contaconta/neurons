function [V0, V0a] = swc_swc2(W, B, v)
%% SWC_CP forms connected components from W
%   V0 = swc_swc2(W,B,v)
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

superRegion = graphtraverse(W,v);
MASK = sparse([],[],[], size(W,1), size(W,1),0);
MASK(superRegion,superRegion) = 1;
W = W.*MASK;

% only operate on the upper-triangle of the adjacencey graph W
W = triu(W);

% make a list of edges in W
Elist = find(W)';


% loop through edges, turn them on/off according to p_e
e_ind = 1; q_e = zeros(size(Elist));
for e = Elist
    [vi, vj] = ind2sub(size(W), e); %#ok<NASGU>
    
    q_e(e_ind) = 1 - exp(-B);  % probability to "turn on" an edge
    
    e_ind = e_ind + 1;
end

r = rand(size(q_e));

Eoff = Elist(r >= q_e);


W(Eoff) = 0;
W = max(W,speye(size(W)));   % add a diagonal so all nodes are self-connected
W = max(W,W');

V0 = graphtraverse(W,v);

MASK = sparse([],[],[], size(W,1), size(W,1),0);
MASK(V0,V0) = 1;

V0a = W.*MASK;
