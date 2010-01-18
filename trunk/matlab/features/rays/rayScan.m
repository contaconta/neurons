function [r c] = rayScan(I, G, angle)

% edge finding parameters
MINPKDIST = 5;
MINPKHEIGHT = 22;   %1.7*mean(gline);
MINAREA = 4;

E = zeros(size(I)); RAY1 = E;

angle = mod(angle,360);
if angle < 0;  angle = angle + 360; end;
% flip the sign of the angle (because matlab image y axis points down)
if angle ~= 0
    angle = 360-angle;
end


%if 0 <= angle

START = [ 0 0 ];
END = [size(I,2)-1 round((size(I,1)-1)*tan(deg2rad(angle)))];    %[c r]
START = START + 1;  END = END + 1;
[c,r] = intline(START(1), END(1), START(2), END(2));



% first, compute the edge map
rup = r - r(length(r));
%rdn = r - r(length(r));
% scan down
while rup(1) <= size(I,1)
    rup = rup + 1;
    valid = rup < size(I,1);
    valid = (rup < size(I,1)) & (rup > 0);
    ra = rup(valid); ca = c(valid);
    
    ind = sub2ind(size(I), ra,ca);
    gline = G(ind);

    if length(gline) > 2
        
        % find the Edge map E by locating peaks in the gradient along the
        % ray
        [locs, pks] = peakfinder(gline); %#ok<NASGU>
        edgeinds = ind(locs);
        E(edgeinds) = 1;
    end
end

keyboard;

% scan up
while rdn(length(rdn)) >= 1
    valid = rdn > 0;
    ra = rdn(valid); ca = c(valid);
    
    ind = sub2ind(size(I), ra,ca);
    gline = G(ind);

    if length(gline) > 2
        [locs, pks] = peakfinder(gline);
        edgeinds = ind(locs);
        E(edgeinds) = 1;
    end
    rdn = rdn - 1;
end





% remove small isolated edges
STATS = regionprops(bwlabel(logical(E)), 'PixelIdxList', 'Area'); %#ok<MRPBW>
for l = 1:length(STATS)
    if STATS(l).Area < MINAREA;
        E(STATS(l).PixelIdxList) = 0;
    end
end
%keyboard;



% second pass, compute the ray features
rup = r;
rdn = rup;
%cflip = flipud(c);
% scan down
while rup(1) <= size(I,1)
    rup = rup + 1;
    valid = rup < size(I,1);
    ra = rup(valid); ca = c(valid);
    
    % traverse the ray to compute the various ray features
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
end



% scan up
while rdn(length(rdn)) >= 1
    valid = rdn > 0;
    ra = rdn(valid); ca = c(valid);
    
    % traverse the ray to compute the various ray features
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
    rdn = rdn - 1;
end









keyboard;

%Sr = round(r); Sc = round(c);




 


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