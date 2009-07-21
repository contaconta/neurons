function intersection = overlap(rectA, rectB)
%
% [x1 y1 x2 y2]
%
% x1 = upper left pixel x location
% y1 = upper left pixel y location
% 
% x2 = first pixel in x direction NOT in box
% y2 = first pixel in y direction NOT in box

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
    %intersection = 1;
    length = abs(xmax - xmin);
    width = abs(ymax - ymin);                     
    intersection = length*width;               
end 



