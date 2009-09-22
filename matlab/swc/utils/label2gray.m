function G = label2gray(L,I)
%% RGB2LABEL converts an RGB image into labeled regions 
%
%   G = label2gray(L,I) creates a grayscale image filling each region in L
%   with the average value of the pixels within that region in I.
%
%   See also LABEL2RGB

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

% fill G with average gray levels
G = zeros(size(L));
Linds = unique(L(:))';
for l=Linds
    CC = bwconncomp(L == l);
    if ~isempty(CC.PixelIdxList)
        g = mean( I(CC.PixelIdxList{1}));
        G(CC.PixelIdxList{1}) = g;
    else
        disp(['could not find any members of l=' num2str(l)]);
    end
end

G = uint8(round(G));