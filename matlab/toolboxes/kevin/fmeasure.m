function fmeas = fmeasure(rectA, rectB)
%
% [x1 y1 x2 y2]
%
%
%
%
%


Ax1 = rectA(1);
Ay1 = rectA(2);
Ax2 = rectA(3);
Ay2 = rectA(4);


Bx1 = rectB(1);
By1 = rectB(2);
Bx2 = rectB(3);
By2 = rectB(4);



xmin = max(Ax1, Bx1);
xmax = min(Ax2, Bx2);
ymin = max(Ay1, By1);
ymax = min(Ay2, By2);


if  ~((xmax > xmin) && (ymax > ymin))
    intersection = 0;
else
    length = abs(xmax - xmin);
    width = abs(ymax - ymin);                     
    intersection = length*width;               
end 

recall = intersection / (abs(Ax2-Ax1)*abs(Ay2-Ay1));
precision = intersection / (abs(Bx2-Bx1)*abs(By2-By1));
fmeas = (2*precision*recall) / (precision + recall);

if isnan(fmeas)
    fmeas = 0;
end