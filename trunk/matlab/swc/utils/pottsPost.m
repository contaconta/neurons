function P = pottsPost(G0, LABELS, B)
%% POTTSPOST
%   P = pottsPost(G0, LABELS)
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

%P = 1;

edges = find(triu(G0) == 1)';
[vi, vj] = ind2sub(size(G0), edges);

P = sum( log(B) * (LABELS(vi) == LABELS(vj)));



% % NON-VECTORIZED version !!!!!
% edges = find(triu(G0) == 1)';
% for e = edges
%     [vi, vj] = ind2sub(size(G0), e);
%     if LABELS(vi) == LABELS(vj)
%         P = P + log(B);
%     end        
% end
