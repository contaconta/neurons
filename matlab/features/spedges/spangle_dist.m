function [SPANGLE, SPNORM] = spangle_dist(I, angle, stride, EDGE)
%SPANGLE_DIST computes a spedge feature in a given direction
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


% we get our edge map based on the index edge_method
%EDGE = edgemethods(I, edge_method);

% compute the gradient information
gh = imfilter(I,fspecial('sobel')' /8,'replicate');
gv = imfilter(I,fspecial('sobel')/8,'replicate');
G(:,:,1) = gv;
G(:,:,2) = gh;
Gmag = sqrt(  sum( (G.^2),3));
Gang = gradientnorm(G);


angvec = unitvector(angle);


warning off MATLAB:nearlySingularMatrix; warning off MATLAB:singularMatrix;
[row, col] = linepoints(I,angle);
warning on MATLAB:nearlySingularMatrix; warning on MATLAB:singularMatrix;

% if there is a top-bottom intersection
SPANGLE = zeros(size(EDGE));
SPNORM = zeros(size(EDGE));

% if the angle is pointing up/down
if ((angle >= 45) && (angle <= 135))  || ((angle >= 225) && (angle <= 315))
    % SCAN TO THE LEFT!
    j = 0;
    colx = col + j;
    inimage = find(colx > 0);
    while ~isempty(inimage);
        rowx = row(inimage);
        colx = col(inimage) + j;
        %lastedge = [rowx(1) colx(1)];  
        lastgrad = angvec';  %squeeze(Gang(rowx(1), colx(1),:));
        lastnorm = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(1) - rowx(i));
                %lastedge = [rowx(i) colx(i)];  
                lastgrad = squeeze(Gang(rowx(i), colx(i),:));
                lastnorm = Gmag(rowx(i), colx(i));
                %lastnorm = norm([gh(rowx(i), colx(i)) gv(rowx(i), colx(i))]);           
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            else
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(1) - rowx(i));
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            end
        end
        j = j-1;
        colx = col + j;
        inimage = find(colx > 0);
    end


    % SCAN TO THE RIGHT!
    j = 1;
    colx = col + j;
    inimage = find(colx <= size(I,2));
    while ~isempty(inimage);
        rowx = row(inimage);
        colx = col(inimage) + j;
        %lastedge = [rowx(1) colx(1)]; 
        lastgrad = angvec';  %squeeze(Gang(rowx(1), colx(1),:));
        lastnorm = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(1) - rowx(i));
                %lastedge = [rowx(i) colx(i)]; 
                lastgrad = squeeze(Gang(rowx(i), colx(i),:));
                lastnorm = Gmag(rowx(i), colx(i));
                %lastnorm = norm([gh(rowx(i), colx(i)) gv(rowx(i), colx(i))]);
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
                
                if (rowx(i)==83 ) && (colx(i) == 67)
                    keyboard;
                end
            else
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(1) - rowx(i));
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            end
        end
        j = j+1;
        colx = col + j;
        inimage = find(colx <= size(I,2));
    end
   
% the angle is pointing left-right (-pi/4 > angle > pi/4) or (3pi/4 > angle > 5pi/4)
else
    % SCAN TO THE bottom!
    j = 0;
    rowx = row + j;
    inimage = find(rowx > 0);
    while ~isempty(inimage);
        rowx = row(inimage) + j;
        colx = col(inimage);
        %lastedge = [rowx(1) colx(1)]; 
        lastgrad = angvec';  %squeeze(Gang(rowx(1), colx(1),:));
        lastnorm = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(2) - colx(i));
                %lastedge = [rowx(i) colx(i)]; 
                lastgrad = squeeze(Gang(rowx(i), colx(i),:));
                lastnorm = Gmag(rowx(i), colx(i));
                %lastnorm = norm([gh(rowx(i), colx(i)) gv(rowx(i), colx(i))]);
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            else
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(2) - colx(i));
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            end
        end
        j = j-1;
        rowx = row + j;
        inimage = find(rowx > 0);
    end


    % SCAN TO THE top!
    j = 1;
    rowx = row + j;
    inimage = find(rowx <= size(I,1));
    while ~isempty(inimage);
        rowx = row(inimage) + j;
        colx = col(inimage);
        %lastedge = [rowx(1) colx(1)]; 
        lastgrad = angvec';  %squeeze(Gang(rowx(1), colx(1),:));
        lastnorm = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(2) - colx(i));
                %lastedge = [rowx(i) colx(i)];
                lastgrad = squeeze(Gang(rowx(i), colx(i),:));
                lastnorm = Gmag(rowx(i), colx(i));
                %lastnorm = norm([gh(rowx(i), colx(i)) gv(rowx(i), colx(i))]);
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            else
                %SPANGLE(rowx(i),colx(i)) = abs(lastedge(2) - colx(i));
                SPANGLE(rowx(i),colx(i)) = angvec * lastgrad;
                SPNORM(rowx(i),colx(i)) = lastnorm;
            end
        end
        j = j+1;
        rowx = row + j;
        inimage = find(rowx <= size(I,1));
    end
end


SPANGLE = SPANGLE(1:stride:size(SPANGLE,1), 1:stride:size(SPANGLE,2));
SPNORM = SPNORM(1:stride:size(SPNORM,1), 1:stride:size(SPNORM,2));


%figure; imagesc(SPANGLE); axis image;





%function d = euclid(x,y)
%d = sum((x-y).^2).^.5;



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
