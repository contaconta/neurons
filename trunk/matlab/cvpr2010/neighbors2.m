function N = neighbors2(A,varargin)
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

if ischar(varargin{length(varargin)})
    % we have been given a index
    i = varargin{1};
else
    % we have been given a sub
    %r = varargin{1};
    %c = varargin{2};
    i = sub2ind(size(A),varargin{1},varargin{2});
end

R = size(A,1);

els = numel(A);
N_offset = -1;
NE_offset = R -1;
E_offset = R;
SE_offset = R + 1;
S_offset = 1;
SW_offset = -R + 1;
W_offset = -R;
NW_offset = -R - 1;

if mod(i,R) == 0
    S_offset = -Inf;
    SE_offset = -Inf;
    SW_offset = -Inf;
end

if mod(i,R) == 1
    N_offset = -Inf;
    NE_offset = -Inf;
    NW_offset = -Inf;    
end


% 4-connectivity
N = [N_offset E_offset S_offset W_offset] +i;

% 8-connectivity
%N = [N_offset NE_offset E_offset SE_offset S_offset SW_offset W_offset NW_offset] + i;

N = N(N>0);
N = N(N<=els);