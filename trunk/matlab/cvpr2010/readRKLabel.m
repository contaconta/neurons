function L = readRKLabel(labelFilenm, s, bitdepth)
%% READRKLABEL reads RK superpixels data files
%   
%   L = readRKLabel(labelFilenm, size) reads a superpixel label data file 
%   from RK and converts it into a matlab label matrix. Size of the image
%   must be specified in SIZE.
%

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


fid = fopen(labelFilenm,'r');

if ~ (strcmp(bitdepth, 'uint16') || strcmp(bitdepth, 'uint32') || strcmp(bitdepth, 'uint8') )
    error('input a valid bitdepth')
end

for i = 1:s(3)
    
    I = fread(fid,[s(2) s(1)], bitdepth);
    L(:,:,i) = I';
end
L = double(L);
L = L+1;

fclose(fid);