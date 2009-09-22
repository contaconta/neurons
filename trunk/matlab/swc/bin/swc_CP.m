function CP = swc_CP(G, B)
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


% only operate on the upper-triangle of the adjacencey graph G
G = triu(G);

% make a list of edges in G
Elist = find(G)';


% loop through edges, turn them on/off according to p_e
e_ind = 1; q_e = zeros(size(Elist));
for e = Elist
    [vi, vj] = ind2sub(size(G), e); %#ok<NASGU>
    
    q_e(e_ind) = 1 - exp(-B);  % probability to "turn on" an edge
    %q_e(e_ind) = KL(e_ind);
    
    e_ind = e_ind + 1;
end

r = rand(size(q_e));

Eoff = Elist(r >= q_e);


% make CP
CP = G;
CP(Eoff) = 0;
CP = max(CP,speye(size(CP)));   % add a diagonal so all nodes are self-connected

% make CP an undirected graph by adding the lower triangle 
CP = max(CP, CP');
