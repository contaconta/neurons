function visualize_haar_feature(feature_string, IMSIZE, varargin)
%VISUALIZE_HAAR_FEATURE
%
%   visualize_haar_feature(feature_string, IMSIZE, IM) plots a haar feature
%   described by feature_string on a standard detector window of size
%   IMSIZE. Optional argument IM passes an image to plot the feature over
%   instead of a blank detector window.

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


%% plot on either a blank image, or over an image passed as optional 3rd argument
if nargin == 3
    A = varargin{1};
    IMSIZE = size(A);
else
    A = .5 * ones(IMSIZE);
end


patterns = {'[W][^_]*', '[B][^_]*'};

for i = 1:length(patterns)
    
    m = regexp(feature_string, patterns(i), 'match');
    
    for j = 1:size(m{1},2)
        m_str = m{1}{j};
    
        ax_str = regexp(m_str, 'ax\w*ay', 'match');
        ax_str2 = regexp(ax_str{1}, '\d*', 'match');
        ax = str2double(ax_str2{1});
        ay_str = regexp(m_str, 'ay\w*bx', 'match');
        ay_str2 = regexp(ay_str{1}, '\d*', 'match');
        ay = str2double(ay_str2{1});
        bx_str = regexp(m_str, 'bx\w*by', 'match');
        bx_str2 = regexp(bx_str{1}, '\d*', 'match');
        bx = str2double(bx_str2{1}) -1;
        by_str = regexp(m_str, 'by\d*', 'match');
        by_str2 = regexp(by_str{1}, '\d*', 'match');
        by = str2double(by_str2{1}) -1;
        %disp([ ax ay bx by]);
        
        if i == 1
            A(ay:by,ax:bx) = 1;   
        else
            A(ay:by,ax:bx) = 0;
        end
    end
end

%pause(0.01);
figure(1);
set(gca, 'Position', [0 0 1 1]);
imshow(A);

