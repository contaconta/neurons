function [intersect_area, intersect_mask, xb, yb] = polyintersect(x1,y1,x2,y2,R,C)


% get the intersection
[xb, yb] = polybool('intersection', x1, y1, x2, y2);

intersect_area = polyarea(xb,yb);
intersect_mask = poly2mask(xb,yb,R,C);


% figure; 
% hold on;
% imagesc(intersect_mask);
% plot(x1,y1,'k-');
% plot(x2,y2,'k-');
% patch(xb,yb, 1, 'FaceColor', 'g');


keyboard;