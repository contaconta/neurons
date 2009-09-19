function L = rgb2label(RGB)
%% RGB2LABEL converts an RGB image into labeled regions 
%
%   L = rgb2label(RGB) creates a label matrix L from an RGB image. Each
%   separate region of RGB with a distinct color is given a label index in
%   L.
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

L = zeros([size(RGB,1) size(RGB,2)]);
colors = squeeze(RGB(1,1,:))';
for r = 1:size(L,1);
    for c = 1:size(L,2);
        
        color =  squeeze(RGB(r,c,:))';
        
        matches = ismember(colors, color);
        matches = matches(:,1) .* matches(:,2) .* matches(:,3);
        l = find(matches, 1);
        if l
            L(r,c) = l;
        else
            colors = [colors ; color]; %#ok<AGROW>
            L(r,c) = size(colors,1);
        end
    end
end

% some colors may have been repeated, we must find them and give new labels
nlabels = max(L(:));
for l=1:nlabels
    CC = bwconncomp(L == l);
    if CC.NumObjects > 1
        for n=2:CC.NumObjects
            L(CC.PixelIdxList{n}) = nlabels + 1;
            nlabels = nlabels + 1;
        end
    end
end