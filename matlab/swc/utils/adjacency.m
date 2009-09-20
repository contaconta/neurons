function [A,Alist] = adjacency(L)
%% NEIGHBORS2 finds neighbor elements in a 2D array
%   A = magic(3)
%   A(neighbors2(A,1,1))  % by sub
%   A(neighbors2(A,8,'ind'))  % by ind
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


% 
% function neighbor_search(L,l, region, neighbors)
% 
% if ~isempty(region)
%     % select the first available region member
%     m = region(1);
% 
%     % remove it from the region list
%     region = region(2:length(region));
% 
%     N = neighbors2(A,m,'sub');
%     neighborlabels = L(N);
%     neighbors = unique([neighbors neighborlabels ~= l]);
%     
%     if ~isempty(N)
%         for n = 1:N
%             [members, region] = neighbor_search(L,l, region, neighbors);
%         end
%     end
% end
% 
