%% IMSEG plots an image with its superpixel segmentation 
%		
%   Usage: [I, E] = imseg(I, L)  where I is the original image (uint8) and L contains
%   the superpixel labels.  Returns I with superpixel boundaries drawn in gray, and E
%   a map of the superpixel boundaries.
%
%   See also NLFILTER

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

function [I, E] = imseg(I, L)

fun = @(x) ~sameval(x(:));
E = nlfilter(L, [3 3], fun);
%E = bwmorph(E, 'shrink');
E = bwmorph(~E, 'thicken', Inf);
E = bwmorph(~E, 'shrink');

I(E == 1) = 50;

