function prettygraph()
%PRETTYGRAPH()
%
%   Prettygraph() changes the look of your graph to a more visually
%   pleasing Economist-style layout, with white horizontal grid lines, 
%   white borders, no axis lines, and a shaded background.
%
%   Example:
%   ================
%   plot(rand(10,1), 'bo-', 'LineWidth', 2);
%   prettygraph;
%
%   See also: PLOT, FIGURE, SAVETOPDF


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


BACKGROUNDCOLOR = [.94 .94 .94];
GRIDWIDTH = 1.75;


a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');

box off
set(a, 'TickLength', [0 0]);

set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
xblock = line(XLim, [YLim(1) YLim(1)], 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
yblock = line([XLim(1) XLim(1)], YLim, 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
set(xblock, 'ZData', [-1 -1]);
set(yblock, 'ZData', [-1 -1]);

l = zeros(size(YTick));
for y = 1:length(YTick)
   l(y) = line(XLim, [YTick(y) YTick(y)], 'LineWidth', GRIDWIDTH, 'Color', [1 1 1]); 
   set(l(y), 'ZData', [-1 -1]);
end

set(gcf, 'Color', [1 1 1]);