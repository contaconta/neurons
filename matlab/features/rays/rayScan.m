function [r c] = rayScan(I, G, angle)

% edge finding parameters
MINPKDIST = 5;
MINPKHEIGHT = 22;   %1.7*mean(gline);
MINAREA = 4;

% allocate arrays for edge E, rays RAY1, ...
E = zeros(size(I)); RAY1 = E;


%% properly format the angle (between 0 and 360 degrees), flip y axis (matlab)
angle = mod(angle,360);
if angle < 0;  angle = angle + 360; end;
if angle ~= 0  % flip the angle over x axis because matlab image y axis points down
    angle = 360-angle;
end

%% create a scanline in direction given by the angle centered in the image
[r c] = scanline(I, angle);

% temp to visualize the scanline
T = E; T(sub2ind(size(I),r,c)) = 1; figure(1); imagesc(T); colormap gray;


%% determine the scanning direction (down or right, depending on angle and image size)
imgangle = rad2deg(atan2(size(E,1), size(E,2)));

if ((angle >= imgangle) && (angle <= 180-imgangle))  || ((angle >= 180 + imgangle) && (angle <= 360-imgangle))
    % scan right
    c = c - max(c) + 1;
    disp('have not handled the scan right case yet');
    keyboard;
    
else
    % scan down
    r = r - max(r) + 1;
    scandown = 1;
end







%% compute the edge map using the gradient G

scanr = r;
%scanr = r - r(length(r));
while min(scanr) <= size(I,1)
    
    % this can be sped up by searching for searching for 1st valid, invalid
    valid = (scanr <= size(I,1)) & (scanr > 0);
    ra = scanr(valid); ca = c(valid);
    
    ind = sub2ind(size(I), ra,ca);
    gline = G(ind);

    if length(gline) > 2
        
        % find the Edge map E by locating peaks in the gradient along the
        % ray
        [locs, pks] = peakfinder(gline); %#ok<NASGU>
        edgeinds = ind(locs);
        E(edgeinds) = 1;
    end
    scanr = scanr + 1;
end

% scanr = r - r(length(r));
% while scanr(1) <= size(I,1)
%     valid = (scanr <= size(I,1)) & (scanr > 0);
%     ra = scanr(valid); ca = c(valid);
%     
%     ind = sub2ind(size(I), ra,ca);
%     gline = G(ind);
% 
%     if length(gline) > 2
%         
%         % find the Edge map E by locating peaks in the gradient along the
%         % ray
%         [locs, pks] = peakfinder(gline); %#ok<NASGU>
%         edgeinds = ind(locs);
%         E(edgeinds) = 1;
%     end
%     scanr = scanr + 1;
% end




%% clean edges -> remove small isolated edges
STATS = regionprops(bwlabel(logical(E)), 'PixelIdxList', 'Area'); %#ok<MRPBW>
for l = 1:length(STATS)
    if STATS(l).Area < MINAREA;
        E(STATS(l).PixelIdxList) = 0;
    end
end





%% compute the ray features
scanr = r;
%scanr = r - r(length(r));
while min(scanr) <= size(I,1)
    valid = (scanr <= size(I,1)) & (scanr > 0);
    ra = scanr(valid); ca = c(valid);
    
    % traverse the scanline (ray) to compute the various ray features
    steps_since_edge = 0;
    for i = length(ra):-1:1
        if E(ra(i),ca(i)) == 1
            steps_since_edge = 0;
            %lastGA = rayVector * [G(r(i),c(i),1); G(r(i),c(i),2)]; 
            %lastGN = GN(r(i),c(i));
        end
        RAY1(ra(i),ca(i)) = steps_since_edge;
        %RAY3(r(i),c(i)) = lastGA;
        %RAY4(r(i),c(i)) = lastGN;
        steps_since_edge = steps_since_edge +1;
    end
    scanr = scanr + 1;
end

figure(2); imagesc(RAY1);
keyboard;






%Sr = round(r); Sc = round(c);


%% ================== SCANLINE ============================================
% SCANLINE returns row, column locations of scanline in direction angle
function [r c] = scanline(I, angle)

% draw a line from 0,0 to the limits of I (+ some extra buffer pixels)
START = [0 0]; 
H = sqrt(size(I,1)^2 + size(I,2)^2);  H = H+(.02*H);
END = round([ H*cosd(angle)   H*sind(angle)]);
START = START + 1;  END = END + 1;
[c,r] = intline(START(1), END(1), START(2), END(2));

% move the line so that it is centered in I
r =  r - round(median(r)) + round(size(I,1)/2);
c =  c - round(median(c)) + round(size(I,2)/2);

% get rid of any r,c that are out of the image boundaries
valid = r > 0 & r <=size(I,1) & c > 0  & c <=size(I,2);
r = r(valid);
c = c(valid);


%% ================== INTLINE =============================================
% INTLINE Integer-coordinate line drawing algorithm.
%   [X, Y] = INTLINE(X1, X2, Y1, Y2) computes an
%   approximation to the line segment joining (X1, Y1) and
%   (X2, Y2) with integer coordinates.  X1, X2, Y1, and Y2
%   should be integers.  INTLINE is reversible; that is,
%   INTLINE(X1, X2, Y1, Y2) produces the same results as
%   FLIPUD(INTLINE(X2, X1, Y2, Y1)).
function [c,r] = intline(c1, c2, r1, r2)

dc = abs(c2 - c1);
dr = abs(r2 - r1);

% Check for degenerate case.
if ((dc == 0) && (dr == 0))
  c = c1;
  r = r1;
  return;
end

flip = 0;
if (dc >= dr)
  if (c1 > c2)
    % Alwars "draw" from left to right.
    t = c1; c1 = c2; c2 = t;
    t = r1; r1 = r2; r2 = t;
    flip = 1;
  end
  m = (r2 - r1)/(c2 - c1);
  c = (c1:c2).';
  r = round(r1 + m*(c - c1));
else
  if (r1 > r2)
    % Alwars "draw" from bottom to top.
    t = c1; c1 = c2; c2 = t;
    t = r1; r1 = r2; r2 = t;
    flip = 1;
  end
  m = (c2 - c1)/(r2 - r1);
  r = (r1:r2).';
  c = round(c1 + m*(r - r1));
end

if (flip)
  c = flipud(c);
  r = flipud(r);
end

% alternative to finding peaks
%[pks, locs] = findpeaks(gline, 'minpeakheight', MINPKHEIGHT, 'minpeakdistance', MINPKDIST);



% if ((0 <= angle) && (angle < 45) ) || (( angle <=315) && (angle < 360))
%     START = [0 size(I,1)-1];  %[X Y] not [r c]
%     END   = [size(I,2)-1  size(I,1)-1 - round((size(I,2)-1)*tan(deg2rad(angle)))];
% elseif (angle <= 45) && (angle < 135)
%     START = [size(I,2)-1 size(I,1)-1];
%     END = [0 size(I,1)-1 - round( (size(I,2)-1)*tan(deg2rad(270-angle))
% elseif (angle <=135) && (angle < 225)
%     Q = 3;
% elseif (angle <=225) && (angle < 315)
%     Q = 4;
% end

% line([START(1) END(1)], [START(2) END(2)]);
% axis([-250 250 -200 200]);
% keyboard;

% if (0 <= angle) && (angle < 90)
%     Q = 1;
% elseif (angle <= 90) && (angle < 180)
%     Q = 2;
% elseif (angle <=180) && (angle < 270)
%     Q = 3;
% else 
%     Q = 4;
% end

% switch Q
%     case 1
%         START = [0 size(I,1)-1 ];
%         END = [size(I,2)-1 size(I,1)-1 - round((size(I,2)-1)*tan(deg2rad(angle)))];
%     case 2
%         START = [size(I,1)-1 size(I,2)-1];
%         END = [0 0 ];  %[ round(    )  0];
%     case 3
%         START = [0 size(I,2)-1]; 
%         END = [0 0];
%     case 4
%         START = [ 0 0 ];
%         END = [size(I,2)-1 -round((size(I,1)-1)*tan(deg2rad(angle)))];    %[c r]
% end
% START = START + 1;  END = END + 1;
% [c,r] = intline(START(1), END(1), START(2), END(2));