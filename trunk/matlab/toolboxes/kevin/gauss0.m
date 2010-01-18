function g = gauss0(sigma)
%% GAUSS0 creates a gaussian kernel of standard deviation 0
%   
%   G = gauss0(SIGMA) returns a 1-dimensional gaussian kernel with standard
%   deviation SIGMA.  Can be used for computing seperable gaussian
%   derivatives.
%
%   Example:
%   -----------------------------
%   gx = gauss0(2);
%   gy = gauss0(2)';  
%   gradh = imfilter(I,gx,'replicate');
%   gradv = imfilter(I,gy,'replicate');
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

hsize = [1 6*sigma+1];   % The filter size.
hsize(2) = ceil(hsize(2));

g = fspecial('gaussian',hsize,sigma);
%g = diff(g);

E = sum(abs(g).^2);
%g = g.*(E);

%E = sum(abs(g).^2);
disp(['Energy: ' num2str(E)]);

% % Magic numbers
% GaussianDieOff = .0001; 
% 
% %PercentOfPixelsNotEdges = .7; % Used for selecting thresholds
% %ThresholdRatio = .4;          % Low thresh is this fraction of the high.
% 
% % Design the filters - a gaussian and its derivative
% 
% pw = 1:30; % possible widths
% ssq = sigma^2;
% width = find(exp(-(pw.*pw)/(2*ssq))> GaussianDieOff ,1,'last');
% if isempty(width)
% width = 1;  % the user entered a really small sigma
% end
% 
% t = (-width:width);
% gau = exp(-(t.*t)/(2*ssq))/(2*pi*ssq);     % the gaussian 1D filter


% 
% 
% %G = gauus0(sigma)
% %
% %Gives a 1-D approximate gaussian filter, G, with an 
% %appropriately sized window length for its standard deviation, sigma.
% 
% g0 = normpdf(-100:100, 0, sigma);
% 
% g0 = g0(find(g0 > .0000001 * max(g0)))';
% 
% 


%%% VERY VERY IMPORTANT GAUSS AND DERIVATIVE %%%%
% g=normpdf([-100:100],0,10);
% plot(g)
% hold on
% g=normpdf([-100:100],0,20);
% plot(g,'r')
% 
% % first derivative of Gaussian
% clf
% len=length(g)
% dg=g(2:len)-g(1:len-1);
% plot(dg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
