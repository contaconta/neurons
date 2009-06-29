%TEST_HAAR
%
%   test function to verify that haar features are being defined properly.
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


clear LEARNERS;

% load an image to display over
I = imread('image.png');
if ~isa(I, 'double')
    cls = class(I);
    I = mat2gray(I, [0 double(intmax(cls))]); 
end

% convert to grasyscale if necessary
if size(I,3) > 1
    I = rgb2gray(I);
end

% specify the types of learners
LEARNERS.types = {'HA_x4_y4_u8_v8'};

% define individual learners
DATASETS.IMSIZE = size(I);
LEARNERS = p_EnumerateLearners(LEARNERS, DATASETS.IMSIZE);

for i = 1:length(LEARNERS.list)
    visualize_haar_feature(LEARNERS.list{i}, DATASETS.IMSIZE, I);
    LEARNERS.list{i}
    
    pause;
end