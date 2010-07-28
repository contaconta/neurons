function filesToMovie(source, filename, fps, filetype)
%   filesToMovie(sourceFolder, movieFilename, fps, filetype)
%
%   Converts a series of images from a SOURCE folder into a high-quality 
%   movie using mencoder. Mplayer/mencoder must be installed on the
%   computer. 
%
%   Example:
%   -----------------------
%   filesToMovie('/home/folder/', 'movie.avi', 24, 'PNG');
%
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

[folder, name, ext, versn] = fileparts(filename); %#ok<NASGU>

fname = [name ext];

if nargin < 4
    filetype = 'PNG';
end
    
switch filetype
    case 'PNG'
        d = dir([source '/*.png']);
    case 'JPG'
        d = dir([source '/*.jpg']);
    case 'TIF'
        d = dir([source '/*.tif']);
    otherwise
        error('unknown file type');
end
    
I = imread([ source '/' d(1).name]);

if nargin < 3
    fps = 12;
end

H = size(I,1);
W = size(I,2);
%T = length(d);

BITRATE = round((50*fps*W*H)/256);


cur_path = pwd;

cd(source);
cmd1 = ['mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=1:"vbitrate=' num2str(BITRATE) ':mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=' num2str(fps) ' -nosound -o /dev/null mf://*.png'];
cmd2 = ['mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=2:"vbitrate=' num2str(BITRATE) ':mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=' num2str(fps) ' -nosound -o ' fname ' mf://*.png'];

system(cmd1);
system(cmd2);

cd(cur_path);

if strcmp(folder, '')
    movefile([source '/' fname], [pwd '/' fname]);
    disp(['movie file written: ' pwd '/' fname ]);
    
else
    movefile([source '/' fname], [folder fname]);
    disp(['movie file written: ' folder '/' fname ]);
end




% clean up

