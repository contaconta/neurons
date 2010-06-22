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

SHOWTICKS = 1;

BACKGROUNDCOLOR = [.92 .92 .92];
FIGBG = [ 1 1 1];
GRIDWIDTH = 1;


a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
XTick = get(a, 'XTick');


YOFFSET = abs(YLim(2)-YLim(1))/85;
%YOFFSETS = 0:YOFFSET/(length(YTick)-1):YOFFSET;
%XYOFFSET = abs(YLim(2)-YLim(1))/20;
set(a, 'Color', BACKGROUNDCOLOR);

box off
if ~SHOWTICKS
    set(a, 'TickLength', [0 0]);
else
    %set(a, 'TickLength', get(a, 'TickLength')*2);
    set(gca, 'TickLength', [.03 .025]);
end
%set(a, 'LineWidth', 0.001);
%xblock = line(XLim, [YLim(1) YLim(1)], 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
%yblock = line([XLim(1) XLim(1)], YLim, 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
%set(xblock, 'ZData', [-1 -1]);
%set(yblock, 'ZData', [-1 -1]);

if strcmp(get(a, 'XScale'), 'log')
    XLim(1) = XTick(1)/10;
    x1 = text(XLim(1), YLim(1),  ['10^{' num2str(round(log10(XLim(1)))) '}']);
    set(x1, 'VerticalAlignment', 'top')
end
if strcmp(get(a, 'YScale'), 'log')
    YLim(1) = YTick(1)/10;
end

l = zeros(size(YTick));
for y = 1:length(YTick)
  l(y) = line(XLim, [YTick(y) YTick(y)], 'LineWidth', GRIDWIDTH, 'Color', [1 1 1]); 
  set(l(y), 'ZData', [-1 -1]);
end


set(a, 'XColor', FIGBG);
set(a, 'YColor', FIGBG);

ya = line([XLim(1) XLim(1)], [YLim(1) YLim(2)]);
set(ya, 'Color', [1 1 1], 'LineWidth', 6);

for i = 1:length(XTick)
    if strcmp(get(a, 'XScale'), 'log')
        x(i) = text(XTick(i), YLim(1),  ['10^{' num2str(round(log10(XTick(i)))) '}']);
    else
        x(i) = text(XTick(i), YLim(1), num2str(XTick(i)));
    end
    set(x(i), 'HorizontalAlignment', 'center');
    set(x(i), 'VerticalAlignment', 'top')
end
for i = 1:length(YTick)
    %y(i) = text(XLim(1), YTick(i)-YOFFSETS(i), num2str(YTick(i)));
    y(i) = text(XLim(1), YTick(i), num2str(YTick(i)));
    set(y(i), 'HorizontalAlignment', 'right');
    set(y(i), 'VerticalAlignment', 'middle')
end


set(gcf, 'Color', FIGBG);

keyboard;
