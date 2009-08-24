function N = frag_normcorr(I, fragment, p)
%FRAG_NORMCORR
%   N = frag_normcorr(I, fragment) computes the normalized correlation
%   between and image I and an image template framgment. The correlation
%   ranges between -1.0 and 1.0
%
%   See also FRAG_FEATURE

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

% I = double(I); fragment = double(fragment);
% 
% mI = mean(I(:));
% mf = mean(fragment(:));
% sI = max(0.0000001, std(I(:)));
% sf = max(0.0000001, std(fragment(:)));
% 
% n = numel(I);
% 
% %keyboard;
% 
% if p==1
%     N = ( 1/(n-1) ) * sum(sum( ((I-mI).*(fragment-mf)) ./ (sI*sf) )) ;
% else
%     N = ( 1/(n-1) ) * sum(sum( (((I-mI).*(fragment-mf)).^p) ./ (sI*sf) )) ;
% end

% TEMPORARILY CONVERT TO A INT32 BECAUSE MEMDAEMON CANNOT STORE DOUBLES!
%N = int32( (N + 1) * (intmax('int32')/2));


N = sum(sum(double(I).*double(fragment)));
