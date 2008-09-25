function A = isinsiderect(r, qlist)
%ISINSIDERECT checks to see if a list of rects is inside of a rect.
%
%   A = isinsiderect(R, RECTLIST) checks to see if a rect or a list of
%   rects falls within the bounds of a rect, R.  Returns:
%       1   if the query rect falls entirely within R, 
%       0.5 if the query rect falls partially within R, 
%       0   if the query rect falls entirely outside of R, and
%      -1   if the query rect contains R
%   Rects have the form q = [XMIN YMIN WIDTH HEIGHT].  To test several
%   rects at once, use cells RECTLIST = {q1, q2, q3, ...}.
%   
%   Example:
%   ------------
%   r  = [97    142   102   143];
%   q1 = [133   159    55    53];   % falls inside of r
%   q2 = [46    103    57    65];   % partially overlaps r
%   q3 = [136    14    78    95];   % falls outside of r
%   A = isinsiderect(r, {q1, q2, q3});
%   find(A > 0)                     % rects which partially or fully overlap r
%   find(A == 1)                    % rects which are fully inside r
%   find(A == 0)                    % rects outside of r
%
%   Copyright 2008 Kevin Smith
%
%   See also IMRECT

if iscell(qlist)
    l = length(qlist);
else
    l = 1;
end
    
for i = 1:l
    
    if iscell(qlist)
        q = qlist{i};
    else
        q = qlist;
    end
    
    
    % LOGICAL TEST    
    X1cond = (r(1) <= q(1)) && (q(1) <= r(1) + r(3)); 
    Y1cond = (r(2) <= q(2)) && (q(2) <= r(2) + r(4));
    X2cond = (r(1) <= q(1) + q(3)) && (r(1) + r(3) >= q(1) + q(3));
    Y2cond = (r(2) <= q(2) + q(4)) && (r(2) + r(4) >= q(2) + q(4));
        
    %disp(['i = ' num2str(i) ',  ' num2str([X1cond X2cond Y1cond Y2cond]) ]);
    
    if X1cond && X2cond && Y1cond && Y2cond
        A(i) = 1;
    elseif X1cond && Y1cond
        A(i) = 0.5;
    elseif X1cond && Y2cond
        A(i) = 0.5;
    elseif X2cond && Y1cond
        A(i) = 0.5;
    elseif X2cond && Y2cond
        A(i) = 0.5;
    elseif q(1) <= r(1) && q(2) <= r(2) && q(1) + q(3) >= r(1) + r(3) && q(2) + q(4) >= r(2) + r(4)
        A(i) = -1;
    else
        A(i) = 0;
    end
end
