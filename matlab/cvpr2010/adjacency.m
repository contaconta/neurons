function [A,Alist] = adjacency(L)
%% NEIGHBORS2 finds neighbor elements in a 2D array
%   [A,Alist] = adjacency(L)
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

lmax = max(L(:));

A = sparse([],[],[],lmax,lmax,0);
Alist = cell(lmax,1);

for l = 1:lmax
    
    region = find(L == l); 
    neighbors = neighbor_search(L,l, region);
   
    % mark the connections in the sparse matrix
    if ~isempty(neighbors)
        A(l,neighbors) = 1;
        A(neighbors,l) = 1;
    
        Alist{l} = neighbors;
    end
end

% add a diagonal so that each node is self-connected
A = max(A, speye(size(A)));



function neighbors = neighbor_search(L,l, region)

neighbors = [];
for m = region'
    
    % get the neighbors of m
    N = neighbors2(L,m,'sub');
    
    % check their labels
    neighborlabels = L(N);
    if find(neighborlabels == 0)
        keyboard;
    end
    neighbors = unique([neighbors neighborlabels(neighborlabels ~= l) ]);
end