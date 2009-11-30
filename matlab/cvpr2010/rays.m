function [RAY1 RAY3 RAY4] = rays(E, G, angle, stride)
%RAYS computes RAY features
%
%   FEATURE = spedge_dist(E, G, ANGLE, STRIDE)  computes a spedge 
%   feature on a grayscale image I at angle ANGLE.  Each pixel in FEATURE 
%   contains the distance to the nearest edge in direction ANGLE.  Edges 
%   are computed using Laplacian of Gaussian zero-crossings (!!! in the 
%   future we may add more methods for generating edges).  SIGMA specifies 
%   the standard deviation of the edge filter.  
%
%   Example:
%   -------------------------
%   I = imread('cameraman.tif');
%   gh = imfilter(I,fspecial('sobel')' /8,'replicate');
%   gv = imfilter(I,fspecial('sobel')/8,'replicate');
%   G(:,:,1) = gv;
%   G(:,:,2) = gh;
%   angle = 30;
%   E = bwmorph(edge(I), 'diag');
%   [Rdist Rori Rnorm] = rays(E, G, angle, 2);
%   imagesc(SPEDGE);  axis image;
%
%   Copyright © 2008 Kevin Smith
%
%   See also SPEDGES, EDGE, VIEW_SPEDGES




% ensure good angles
angle = mod(angle,360);
if angle < 0;  angle = angle + 360; end;

% compute the gradient norm GN
G = double(G);
GN = sqrt(sum((G.^2),3));

% convert the Gradient G into unit vectors
G = gradientnorm(G);



% get a scanline in direction angle
warning off MATLAB:nearlySingularMatrix; warning off MATLAB:singularMatrix;
[Sr, Sc] = linepoints(E,angle);
warning on MATLAB:nearlySingularMatrix; warning on MATLAB:singularMatrix;

% initialize the output matrices
RAY1 = zeros(size(E));  % distrance rays
RAY3 = zeros(size(E));  % gradient orientation
RAY4 = zeros(size(E));  % gradient norm

% determine the unit vector in the direction of the Ray
rayVector = unitvector(angle);

% if S touches the top & bottom of the image
if ((angle >= 45) && (angle <= 135))  || ((angle >= 225) && (angle <= 315))
    % SCAN TO THE LEFT!
    j = 0;
    c = Sc + j;
    inimage = find(c > 0);
    while ~isempty(inimage);
        r = Sr(inimage);
        c = Sc(inimage) + j;
        steps_since_edge = 0;  % the border of the image serves as an edge
        lastGN = 0;
        lastGA = 1;
        for i = 1:length(r);
            if E(r(i),c(i)) == 1
                steps_since_edge = 0;
                lastGA = rayVector * [G(r(i),c(i),1); G(r(i),c(i),2)]; 
                %if isnan(lastGA); keyboard; end;
                lastGN = GN(r(i),c(i));
            end
            RAY1(r(i),c(i)) = steps_since_edge;
            RAY3(r(i),c(i)) = lastGA;
            RAY4(r(i),c(i)) = lastGN;
            steps_since_edge = steps_since_edge +1;
        end
        j = j-1;
        c = Sc + j;
        inimage = find(c > 0);
    end


    % SCAN TO THE RIGHT!
    j = 1;
    c = Sc + j;
    inimage = find(c <= size(E,2));
    while ~isempty(inimage);
        r = Sr(inimage);
        c = Sc(inimage) + j;
        steps_since_edge = 0;  % the border of the image serves as an edge
        lastGN = 0;
        lastGA = 1;
        for i = 1:length(r);
            if E(r(i),c(i)) == 1
                steps_since_edge = 0;
                lastGA = rayVector * [G(r(i),c(i),1); G(r(i),c(i),2)]; 
                %if isnan(lastGA); keyboard; end;
                lastGN = GN(r(i),c(i));
            end
            RAY1(r(i),c(i)) = steps_since_edge;
            RAY3(r(i),c(i)) = lastGA;
            RAY4(r(i),c(i)) = lastGN;
            steps_since_edge = steps_since_edge +1;
        end
        j = j+1;
        c = Sc + j;
        inimage = find(c <= size(E,2));
    end
   
% if S touches left & right of image (-pi/4 > angle > pi/4) or (3pi/4 > angle > 5pi/4)
else
    % SCAN TO THE bottom!
    j = 0;
    r = Sr + j;
    inimage = find(r > 0);
    while ~isempty(inimage);
        r = Sr(inimage) + j;
        c = Sc(inimage);
        steps_since_edge = 0;  % the border of the image serves as an edge
        lastGN = 0;
        lastGA = 1;
        for i = 1:length(r);
            if E(r(i),c(i)) == 1
                steps_since_edge = 0;
                lastGA = rayVector * [G(r(i),c(i),1); G(r(i),c(i),2)]; 
                %if isnan(lastGA); keyboard; end;
                lastGN = GN(r(i),c(i));
            end
            RAY1(r(i),c(i)) = steps_since_edge;
            RAY3(r(i),c(i)) = lastGA;
            RAY4(r(i),c(i)) = lastGN;
            steps_since_edge = steps_since_edge +1;
        end
        j = j-1;
        r = Sr + j;
        inimage = find(r > 0);
    end


    % SCAN TO THE top!
    j = 1;
    r = Sr + j;
    inimage = find(r <= size(E,1));
    while ~isempty(inimage);
        r = Sr(inimage) + j;
        c = Sc(inimage);
        steps_since_edge = 0;  % the border of the image serves as an edge
        lastGN = 0;
        lastGA = 1;
        for i = 1:length(r);
            if E(r(i),c(i)) == 1
                steps_since_edge = 0;
                lastGA = rayVector * [G(r(i),c(i),1); G(r(i),c(i),2)]; 
                %if isnan(lastGA); keyboard; end;
                lastGN = GN(r(i),c(i));
            end
            RAY1(r(i),c(i)) = steps_since_edge;
            RAY3(r(i),c(i)) = lastGA;
            RAY4(r(i),c(i)) = lastGN;
            steps_since_edge = steps_since_edge +1;
        end
        j = j+1;
        r = Sr + j;
        inimage = find(r <= size(E,1));
    end
end

if stride ~= 1
    RAY1 = RAY1(1:stride:size(RAY1,1), 1:stride:size(RAY1,2));
    RAY3 = RAY3(1:stride:size(RAY3,1), 1:stride:size(RAY3,2));
    RAY4 = RAY4(1:stride:size(RAY4,1), 1:stride:size(RAY4,2));
end





function [Sr, Sc] = linepoints(E,Angle)
% defines the points in a line in an image at an arbitrary angle
%
%
%
%


% flip the sign of the angle (matlab y axis points down for images) and
% convert to radians
if Angle ~= 0
    angle = deg2rad(360 - Angle);
else
    angle = Angle;
end

% format the angle so it is between 0 and less than pi/2
if angle > pi; angle = angle - pi; end
if angle == pi; angle = 0; end


% find where the line intercepts the edge of the image.  draw a line to
% this point from (1,1) if 0<=angle<=pi/2.  otherwise pi/2>angle>pi draw 
% from the upper left corner down.  linex and liney contain the points of 
% the line

%keyboard;
if (angle >= 0 ) && (angle <= pi/2)
    START = [1 1]; 
    A_bottom_intercept = [-tan(angle) 1; 0 1];  B_bottom_intercept = [0; size(E,1)-1];
    A_right_intercept  = [-tan(angle) 1; 1 0];  B_right_intercept  = [0; size(E,2)-1];
    bottom_intercept = round(A_bottom_intercept\B_bottom_intercept);
    right_intercept  = round(A_right_intercept\B_right_intercept);

    if right_intercept(2) <= size(E,1)-1
        END = right_intercept + [1; 1];
    else
        END = bottom_intercept + [1; 1];
    end
    [linex,liney] = intline(START(1), END(1), START(2), END(2));
else
    START = [1, size(E,1)];
    A_top_intercept = [tan(pi - angle) 1; 0 1];  B_top_intercept = [size(E,1); 1];
    A_right_intercept  = [tan(pi - angle) 1; 1 0];  B_right_intercept  = [size(E,1); size(E,2)-1];
    top_intercept = round(A_top_intercept\B_top_intercept);
    right_intercept  = round(A_right_intercept\B_right_intercept);

    if (right_intercept(2) < size(E,1)-1) && (right_intercept(2) >= 1)
        END = right_intercept + [1; 0];
    else
        END = top_intercept + [1; 0];
    end
    [linex,liney] = intline(START(1), END(1), START(2), END(2));
end

Sr = round(liney); Sc = round(linex);

% if the angle points to quadrant 1 or 4, we need to re-sort the elements 
% of Sr and Sc so they increase in the direction of the angle

if (270 <= Angle) || (Angle < 90)
    %reverse_inds = length(Sr):-1:1;
    %Sr = Sr(reverse_inds);
    %Sc = Sc(reverse_inds);
    Sr = flipud(Sr);
    Sc = flipud(Sc);
end




function [x,y] = intline(x1, x2, y1, y2)
% intline creates a line between two points
%INTLINE Integer-coordinate line drawing algorithm.
%   [X, Y] = INTLINE(X1, X2, Y1, Y2) computes an
%   approximation to the line segment joining (X1, Y1) and
%   (X2, Y2) with integer coordinates.  X1, X2, Y1, and Y2
%   should be integers.  INTLINE is reversible; that is,
%   INTLINE(X1, X2, Y1, Y2) produces the same results as
%   FLIPUD(INTLINE(X2, X1, Y2, Y1)).

dx = abs(x2 - x1);
dy = abs(y2 - y1);

% Check for degenerate case.
if ((dx == 0) && (dy == 0))
  x = x1;
  y = y1;
  return;
end

flip = 0;
if (dx >= dy)
  if (x1 > x2)
    % Always "draw" from left to right.
    t = x1; x1 = x2; x2 = t;
    t = y1; y1 = y2; y2 = t;
    flip = 1;
  end
  m = (y2 - y1)/(x2 - x1);
  x = (x1:x2).';
  y = round(y1 + m*(x - x1));
else
  if (y1 > y2)
    % Always "draw" from bottom to top.
    t = x1; x1 = x2; x2 = t;
    t = y1; y1 = y2; y2 = t;
    flip = 1;
  end
  m = (x2 - x1)/(y2 - y1);
  y = (y1:y2).';
  x = round(x1 + m*(y - y1));
end
  
if (flip)
  x = flipud(x);
  y = flipud(y);
end


function v = gradientnorm(v)
v = double(v); eta = .00000001;
mag = sqrt(  sum( (v.^2),3)) + eta;
v = v ./ repmat(mag, [1 1 2]);

