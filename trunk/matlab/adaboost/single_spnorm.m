function [F, EDGE, G, gh, gv] = single_spnorm(angle, stride, edge_method, r, c, LN, I, varargin)
%SPEDGE_DIST computes a spedge feature in a given direction
%
%   FEATURE = spedge_dist(I, ANGLE, STRIDE, EDGE_METHOD)  computes a spedge 
%   feature on a grayscale image I at angle ANGLE.  Each pixel in FEATURE 
%   contains the distance to the nearest edge in direction ANGLE.  Edges 
%   are computed using Laplacian of Gaussian zero-crossings (!!! in the 
%   future we may add more methods for generating edges).  SIGMA specifies 
%   the standard deviation of the edge filter.  
%
%   Example:
%   -------------------------
%   I = imread('cameraman.tif');
%   SPEDGE = spedge_dist(I,30,2);
%   imagesc(SPEDGE);  axis image;
%
%   Copyright © 2008 Kevin Smith
%
%   See also SPEDGES, EDGE, VIEW_SPEDGES

%% build a binary image containing edges of I

angle = mod(angle,360);
if angle < 0;  angle = angle + 360; end;

% if we are provided with the EDGE image, we can skip computing it
if nargin > 6
    EDGE = I;
    G = varargin{1};
    gh = varargin{2};
    gv = varargin{3};
else
    EDGE = edgemethods(I, edge_method);
    % compute the gradient information
    gh = imfilter(I,fspecial('sobel')' /8,'replicate');
    gv = imfilter(I,fspecial('sobel')/8,'replicate');
    G(:,:,1) = gv;
    G(:,:,2) = gh;
    G = gradientnorm(G);
end

if isempty(LN)
    warning off MATLAB:nearlySingularMatrix; warning off MATLAB:singularMatrix;
    [row, col] = linepoints(I,angle);
    warning on MATLAB:nearlySingularMatrix; warning on MATLAB:singularMatrix;
else
    ang_ind = find(LN(1).angles == angle, 1);
    row = LN(ang_ind).r;
    col = LN(ang_ind).c;
end




%% step 1:  align rowx colx with your scan point
% make r and c fit adjust for the stride
r = 1 + (r-1)*stride;
c = 1 + (c-1)*stride;

% if the angle is pointing up/down
if ((angle >= 45) && (angle <= 135))  || ((angle >= 225) && (angle <= 315))
    cdiff = (c - col(row == r));
    if cdiff > 0
        col = col + cdiff;
        inimage = find(col <= size(I,2));
    else
        col = col + cdiff;
        inimage = (col > 0);
    end
    col = col(inimage);
    row = row(inimage);
else
    rdiff = (r - row(col == c));
    if rdiff > 0
        row = row + rdiff;
        inimage = find(row <= size(I,1));
    else
        row = row + rdiff;
        inimage = (row > 0);
    end
    col = col(inimage);
    row = row(inimage);
end


%% step 2: scan until we get to the point (r,c)
%lastedge = [row(1) col(1)];
%lastgrad = angvec';
lastnorm = 0;

for i = 1:length(row)
    
    if EDGE(row(i),col(i)) == 1
        lastnorm = norm([gh(row(i), col(i)) gv(row(i), col(i))]);
        %lastgrad = squeeze(G(row(i), col(i),:));
        %lastedge = [row(i) col(i)];
    end
    if isequal([r c], [row(i) col(i)])
        F = lastnorm;
        %F = angvec * lastgrad;
        %F = abs(lastedge(1) - row(i));
        break
    end
end



% % if the angle is pointing up/down
% if ((angle >= 45) && (angle <= 135))  || ((angle >= 225) && (angle <= 315))
%     for i = 1:length(row)
%         if EDGE(row(i),col(i)) == 1
%             lastgrad = squeeze(G(row(i), col(i),:));
%             %lastedge = [row(i) col(i)];
%         end
%         if isequal([r c], [row(i) col(i)])
%             
%             %F = abs(lastedge(1) - row(i));
%             F = angvec * lastgrad;
%             break
%         end
%         
%     end
% else
%     for i = 1:length(col)
%          if EDGE(row(i),col(i)) == 1
%             lastnorm = norm([gh(row(i), col(i)) gv(row(i), col(i))]);
%             %lastgrad = squeeze(G(row(i), col(i),:));
%             %lastedge = [row(i) col(i)];
%         end
%         if isequal([r c], [row(i) col(i)])
%             %F = abs(lastedge(2) - col(i));
%             F = lastnorm;
%             break
%         end
%        
%     end
% end




function [row, col] = linepoints(I,Angle)
%
% defines the points in a line in an image at an arbitrary angle
%
%
%
%


% flip the sign of the angle (matlab y axis points down for images) and
% convert to radians
if Angle ~= 0
    %angle = deg2rad(Angle);
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
if (angle >= 0 ) && (angle <= pi/2)
    START = [1 1]; 
    A_bottom_intercept = [-tan(angle) 1; 0 1];  B_bottom_intercept = [0; size(I,1)-1];
    A_right_intercept  = [-tan(angle) 1; 1 0];  B_right_intercept  = [0; size(I,2)-1];
    bottom_intercept = round(A_bottom_intercept\B_bottom_intercept);
    right_intercept  = round(A_right_intercept\B_right_intercept);

    if right_intercept(2) <= size(I,1)-1
        END = right_intercept + [1; 1];
    else
        END = bottom_intercept + [1; 1];
    end
    [linex,liney] = intline(START(1), END(1), START(2), END(2));
else
    START = [1, size(I,1)];
    A_top_intercept = [tan(pi - angle) 1; 0 1];  B_top_intercept = [size(I,1); 1];
    A_right_intercept  = [tan(pi - angle) 1; 1 0];  B_right_intercept  = [size(I,1); size(I,2)-1];
    top_intercept = round(A_top_intercept\B_top_intercept);
    right_intercept  = round(A_right_intercept\B_right_intercept);

    if (right_intercept(2) < size(I,1)-1) && (right_intercept(2) >= 1)
        END = right_intercept + [1; 0];
    else
        END = top_intercept + [1; 0];
    end
    [linex,liney] = intline(START(1), END(1), START(2), END(2));
end

row = round(liney); col = round(linex);

% if the angle points to quadrant 2 or 3, we need to re-sort the elements 
% of row and col so they increase in the direction of the angle

if (270 <= Angle) || (Angle < 90)
    reverse_inds = length(row):-1:1;
    row = row(reverse_inds);
    col = col(reverse_inds);
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
