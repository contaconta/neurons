function F = frag_feature(I, fragment, BB, p)
%FRAG_NORMCORR
%   F = frag_normcorr(I, fragment, BB, p) computes the informative feature
%   fragement response of a FRAGMENT to image I. The fragment is scaled and
%   placed according to the bounding box BB within a mask of size I.  BB is 
%   a 4-element vector with the form [XMIN YMIN WIDTH HEIGHT]; these values 
%   are specified in spatial coordinates, so the size of the scaled 
%   fragment will be WIDTH+1 HEIGHT+1. Feature value is the normalized
%   correleation between the patch and framgment. The correlation
%   ranges between -1.0 and 1.0. P is an exponent for sensitivity (1 = low
%   for textures, 10 = highly specific).
%
%   See also FRAG_NORMCORR

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


% extract a patch from I defined by the BB
patch = I(BB(2):BB(2)+BB(4), BB(1):BB(1)+BB(3));

% scale the fragment to the appropriate size, if necessary
if ~isequal(size(fragment), [BB(4)+1 BB(3)+1])
    fragment = imresize(fragment, [BB(4)+1 BB(3)+1]);
end

if isequal(size(patch), size(fragment))
    
    % the feature response is the normalized correlation between the patch
    % and the scaled fragment
    F = frag_normcorr(patch, fragment, p);
    
else
   error('Error frag_feature: the size of the patch extracted with the BB did not match the size of the scaled fragment');
end
