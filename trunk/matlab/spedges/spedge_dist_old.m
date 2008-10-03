function FEAT = spedge_dist(I, angle, sigma)
%
%
%
%
%
%
%
%

%% build a binary image containing edges of I

angle = mod(angle,360);
if angle < 0;  angle = angle + 360; end;


% we use a zero-crossing laplacian of gaussian to ensure closed contours
EDGE = edge(I, 'log', 0, sigma);

warning off MATLAB:nearlySingularMatrix; warning off MATLAB:singularMatrix;
[row, col] = linepoints(I,angle);
warning on MATLAB:nearlySingularMatrix; warning on MATLAB:singularMatrix;


% if there is a top-bottom intersection
FEAT = zeros(size(EDGE));

% if the angle is pointing up/down
if ((angle >= 45) && (angle <= 135))  || ((angle >= 225) && (angle <= 315))
    % SCAN TO THE LEFT!
    j = -1;
    colx = col + j;
    inimage = find(colx > 0);
    while ~isempty(inimage);
        rowx = row(inimage);
        colx = col(inimage) + j;
        prev = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                FEAT(rowx(i),colx(i)) = 0;
            else
                FEAT(rowx(i), colx(i)) = prev + 1;
            end
            prev = FEAT(rowx(i), colx(i));
        end
        j = j-1;
        colx = col + j;
        inimage = find(colx > 0);
        %keyboard;
    end


    % SCAN TO THE RIGHT!
    j = 1;
    colx = col + j;
    inimage = find(colx <= size(I,2));
    while ~isempty(inimage);
        rowx = row(inimage);
        colx = col(inimage) + j;
        prev = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                FEAT(rowx(i),colx(i)) = 0;
            else
                FEAT(rowx(i), colx(i)) = prev + 1;
            end
            prev = FEAT(rowx(i), colx(i));
        end
        j = j+1;
        colx = col + j;
        inimage = find(colx <= size(I,2));
        %keyboard;
    end
   
% the angle is pointing left-right
else
      % SCAN TO THE bottom!
    j = -1;
    rowx = row + j;
    inimage = find(rowx > 0);
    while ~isempty(inimage);
        rowx = row(inimage) + j;
        colx = col(inimage);
        prev = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                FEAT(rowx(i),colx(i)) = 0;
            else
                FEAT(rowx(i), colx(i)) = prev + 1;
            end
            prev = FEAT(rowx(i), colx(i));
        end
        j = j-1;
        rowx = row + j;
        inimage = find(rowx > 0);
        %keyboard;
    end


    % SCAN TO THE top!
    j = 1;
    rowx = row + j;
    inimage = find(rowx <= size(I,1));
    while ~isempty(inimage);
        rowx = row(inimage) + j;
        colx = col(inimage);
        prev = 0;
        for i = 1:length(rowx);
            if EDGE(rowx(i),colx(i)) == 1
                FEAT(rowx(i),colx(i)) = 0;
            else
                FEAT(rowx(i), colx(i)) = prev + 1;
            end
            prev = FEAT(rowx(i), colx(i));
        end
        j = j+1;
        rowx = row + j;
        inimage = find(rowx <= size(I,1));
        %keyboard;
    end
end






% FEAT = double(~EDGE);
% % FEAT = zeros(size(EDGE));
% prev = 0;
% for i = 1:length(row);
%     if EDGE(row(i),col(i)) == 1
%         %disp('edge')
%         FEAT(row(i),col(i)) = 0;
%     else
%         %disp('no edge')
%         FEAT(row(i), col(i)) = prev + 1;
%     end
%     prev = FEAT(row(i), col(i));
% %     FEAT(row(i),col(i))
% end
%     
figure; imagesc(FEAT); axis image;
% 
% A = zeros(size(I));
% for i = 1:length(row)
%     A(row(i), col(i)) = i; 
% end
% figure; imagesc(A > 0); axis image;

%figure; imagesc(EDGE); axis image;


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

if (90 <= Angle) && (Angle < 180)
    reverse_inds = length(row):-1:1;
    row = row(reverse_inds);
    col = col(reverse_inds);
elseif (180 <= Angle) && (Angle < 270)
    reverse_inds = length(row):-1:1;
    row = row(reverse_inds);
    col = col(reverse_inds);
end

% if (0 <= Angle) && (Angle < 90)
%     disp('okay! 0<=Angle<90');
% elseif (90 <= Angle) && (Angle < 180)
%     reverse_inds = [length(row):-1:1];
%     row = row(reverse_inds);
%     col = col(reverse_inds);
%     disp('dunno! 90<=Angle<180');
% elseif (180 <= Angle) && (Angle < 270)
%     reverse_inds = [length(row):-1:1];
%     row = row(reverse_inds);
%     col = col(reverse_inds);
%     disp('dunno! 180<=Angle<270');
% else
%     disp('dunno! 270<=Angle<360');
% end






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
