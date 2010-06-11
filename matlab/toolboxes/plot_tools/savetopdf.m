function savetopdf(varargin)
%SAVETOPDF()
%
%   Savetopdf(filename) saves the current plot to a pdf file and crops it
%   so that any extra whitespace is removed. Savetopdf(H, filename) saves
%   the figure with handle H to a file. Requires PDFCROP linux utility by
%   Heiko Oberdiek.
%
%   Example:
%   ================
%   plot(rand(10,1), 'bo-', 'LineWidth', 2);
%   prettygraph;
%   savetopdf('figure1.pdf');
%
%   See also: PLOT, FIGURE, PRETTYGRAPH


%   Copyright © 2010 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Author:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

for i = 1:nargin
    switch class(varargin{i})    
        case 'char'
            filename = varargin{1};
        otherwise
            h = varargin{i};
    end
end

if ~exist('h', 'var')
    h = gcf;
end
if ~exist('filename', 'var')
    error('filename not specified');
end


[pathstr, name, ext] = fileparts(filename); %#ok<NASGU>
if ~exist(pathstr, 'dir')
    mkdir(pathstr);
end

set(h, 'InvertHardcopy', 'off');

% SAVE THE OUTPUT TO EPS, CONVERT TO PDF AND CROP!
saveas(h, filename, 'pdf');
cmdcrop = ['pdfcrop ' filename ' ' filename  ];
system(cmdcrop);

% keyboard;
% cmdrot = ['pdftk ' filename ' cat 1W output tempxxx.pdf'];
% cmdmv = ['mv tempxxx.pdf ' filename];
% system(cmdrot);
% system(cmdmv);

