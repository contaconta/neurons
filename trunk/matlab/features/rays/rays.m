function [RAY1 RAY3 RAY4] = rays(E, gv, gh, angle, varargin)
%RAYS extracts Ray features [dist, orientation, norm]
%
%   [Rdist Rori Rnorm] = rays(E, GV, GH, angle) computes ray features in the
%   direction given by ANGLE, given an input edge map E and the horizontal 
%   gradient GH and vertical gradient GV. Outputs Rdist contains type 1 Ray
%   distance features, Rori contains type 3 orientation features, and Rnorm
%   contains type 4 norm features. Type 2 Ray difference features can be
%   computed from type 1 features.  See example below.
%
%   Example (compute Rays at 30 degrees):
%   -------------------------------------
%   I = imread('coins.png');
%   gh = imfilter(I,fspecial('sobel')' /8,'replicate');
%   gv = imfilter(I,fspecial('sobel')/8,'replicate');
%   E = bwmorph(edge(I, 'canny', .5,1), 'diag');
%   [Rdist Rori Rnorm] = rays(E, gv, gh, 30); imagesc(Rdist); axis image;
%
%   Example (Rays extracted at multiple orientations):
%   --------------------------------------------------
%   angles = 0:30:330; 
%   Rdist = zeros([size(I,1) size(I,2) length(angles)]);
%   Rori = Rdist; Rnorm = Rdist;
%   for i = 1:length(angles)
%       [Rdist(:,:,i) Rori(:,:,i) Rnorm(:,:,i)] = rays(E, gv, gh, angles(i));
%       imagesc(Rdist(:,:,i)); axis image; drawnow; refresh; 
%   end
%
%   Example (re-orient Rays to be rotationally invariant by shifting them 
%            so orientation with longest ray becomes 0 degrees, SLOW):
%   ------------------------------------------------------------------
%   for r = 1:size(I,1)
%       for c = 1:size(I,2)
%           shift_places = -find(Rdist(r,c,:) == max(Rdist(r,c,:)),1)+1;
%           a = 1:size(Rdist,3);
%           a = circshift(a, [0 shift_places]);
%           Rdist(r,c,:) = Rdist(r,c,a);
%           Rori(r,c,:)  = Rori(r,c,a);
%           Rnorm(r,c,:) = Rnorm(r,c,a);
%       end
%   end
%
%   Example (compute type 2 difference features from Type 1 features):
%   ------------------------------------------------------------------
%   pairs = combnk(angles, 2); eta = 1;
%   for c = 1:size(pairs,1);
%       angle1 = angles == pairs(c,1);
%       angle2 = angles == pairs(c,2);
%       Rdiff = (Rdist(:,:,angle1) - Rdist(:,:,angle2)) ./ (Rdist(:,:,angle1)+eta);
%       Rdiff = exp(Rdiff);     % exponential difference can be more informative
%       imagesc(Rdiff); axis image; drawnow; refresh; 
%       if c == 6; disp('with rot inv: difference between longest ray and its opposite direction'); pause(3); end;
%   end
%
%   Copyright © 2009 Kevin Smith
%
%   See also EDGE, IMFILTER, FSPECIAL


% ensure valid angles (between 0 and 360)
angle = mod(angle,360);
if angle < 0;  angle = angle + 360; end;

% fill G with the vertical and horizontal gradient images
G(:,:,1) = double(gv);  G(:,:,2) = double(gh);

% compute the gradient norm GN
GN = sqrt(sum((G.^2),3));

% convert the gradient images in G into unit vectors
G = gnormalize(G);

% get a scanline in direction 'angle'
warning off MATLAB:nearlySingularMatrix; warning off MATLAB:singularMatrix;
[Sr, Sc] = scanline(E,angle);
warning on MATLAB:nearlySingularMatrix; warning on MATLAB:singularMatrix;

% allocate output matrices
RAY1 = zeros(size(E));  % distrance rays
RAY3 = zeros(size(E));  % gradient orientation
RAY4 = zeros(size(E));  % gradient norm

% determine the unit vector in the direction of the Ray
rayVector = unitvector(angle);

%% Main loop: compute rays along scanline, adjust scanline, iterate

imgangle = rad2deg(atan2(size(E,1), size(E,2)));

% if S touches the top & bottom of the image
if ((angle >= imgangle) && (angle <= 180-imgangle))  || ((angle >= 180 + imgangle) && (angle <= 360-imgangle))
    
    % SCAN LEFT
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


    % SCAN RIGHT
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
    % SCAN TO BOTTOM
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

    % SCAN TO TOP
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

% specfying the stride will downsample the ray feature image
if nargin > 4
    stride = varargin{1};
    RAY1 = RAY1(1:stride:size(RAY1,1), 1:stride:size(RAY1,2));
    RAY3 = RAY3(1:stride:size(RAY3,1), 1:stride:size(RAY3,2));
    RAY4 = RAY4(1:stride:size(RAY4,1), 1:stride:size(RAY4,2));
end



%% SUB-FUNCTIONS appear below ---------------------------------------------


%% =============== SCANLINE ===============================================
% [Sr, Sc] = scanline(E,Angle)
%   Creates a prototype scanline in direction Angle, fit to the image. Sr
%   constains the row coordinates of the scanline, Sc contains the column
%   coordinates of the scanline.
function [Sr, Sc] = scanline(E,Angle)

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
    Sr = flipud(Sr);
    Sc = flipud(Sc);
end


%% ================== INTLINE =============================================
% INTLINE Integer-coordinate line drawing algorithm.
%   [X, Y] = INTLINE(X1, X2, Y1, Y2) computes an
%   approximation to the line segment joining (X1, Y1) and
%   (X2, Y2) with integer coordinates.  X1, X2, Y1, and Y2
%   should be integers.  INTLINE is reversible; that is,
%   INTLINE(X1, X2, Y1, Y2) produces the same results as
%   FLIPUD(INTLINE(X2, X1, Y2, Y1)).
function [x,y] = intline(x1, x2, y1, y2)

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

%% ============== Gnormalize ============================================
% N = gnormalize(G)
%   transforms a NxMx2 array containing the x-gradient and y-gradient of an
%   image to unit vectors.
function v = gnormalize(v)
v = double(v); eta = .00000001;
mag = sqrt(  sum( (v.^2),3)) + eta;
v = v ./ repmat(mag, [1 1 2]);

%% ============== UNITVECTOR ==============================================
% unitvec = unitvector(angle)
% unitvec = unitvector(vector)
%   u = unitvector(35);  % can accept angle arguments in degrees
%   u = unitvector([2 5]);  % can accept vector arguments
function unitvec = unitvector(data)

% we have been given an angle
if length(data) == 1   
    unitvec(1) = sind(data);
    unitvec(2) = cosd(data);
% we have been given a vector    
else 
    % ensure it is a column vector
    data = data(:);
    if isequal(data, [0 0]');
        % we this has no magnitude, pick a random angle
        ang = rand(1)*2*pi;
        unitvec(1) = sin(ang);
        unitvec(2) = cos(ang);
    else
        % normalize the vector
        unitvec = squeeze(l2norm(data));
    end
end
