function [xpoly, ypoly] = line2poly(x,y, w)

% the polygon width w has a default of 2 if not specified
if ~exist('w', 'var')
    w = 2;
end
l = w / 2;



% figure; hold on;
% plot(x,y,'ks');


for i = 1:numel(x)-1
    
    x1 = x(i);
    x2 = x(i+1);
    y1 = y(i);
    y2 = y(i+1);
    c = [ (x1+x2)/2  (y1+y2)/2 ];
    
    angle = atan2d( (y2-y1) , (x2 - x1));
    angle_perp_a = angle + 90;
    angle_perp_b = angle - 90;
    
    
    ang1 = atan2d(  (y1-c(2)) ,  (x1-c(1)) );
    x1o = x1 + l * cosd(ang1);
    y1o = y1 + l * sind(ang1);
    ang2 = ang1 + 180;
    x2o = x2 + l * cosd(ang2);
    y2o = y2 + l * sind(ang2);

%     plot(c(1), c(2), 'mx');
%     plot([x1 x1o], [y1 y1o], 'c-');
%     plot([x2 x2o], [y2 y2o], 'y-');
%     text(c(1),c(2), sprintf('segment %d', i));
    
    x1a = x1o + w * cosd(angle_perp_a);
    y1a = y1o + w * sind(angle_perp_a);
    x1b = x1o + w * cosd(angle_perp_b);
    y1b = y1o + w * sind(angle_perp_b);
    
    x2a = x2o + w * cosd(angle_perp_a);
    y2a = y2o + w * sind(angle_perp_a);
    x2b = x2o + w * cosd(angle_perp_b);
    y2b = y2o + w * sind(angle_perp_b);
    
    [xcw,ycw] = poly2cw([x1a x1b x2b x2a],[y1a y1b y2b y2a]);
%     xcw = [x1a x1b x2b x2a];
%     ycw = [y1a y1b y2b y2a];

    p(i).x = xcw;
    p(i).y = ycw;
end



union.x = p(1).x;
union.y = p(1).y;
for i = 1:numel(p)-1    
    [xa, ya] = polybool('union', union.x, union.y, p(i+1).x, p(i+1).y);
    [xa,ya] = poly2cw(xa,ya);
    union.x = xa;
    union.y = ya;
end
 
xpoly = union.x;
ypoly = union.y;

% h = patch(union.x,union.y, 1, 'FaceColor', 'k', 'FaceAlpha', .5);

