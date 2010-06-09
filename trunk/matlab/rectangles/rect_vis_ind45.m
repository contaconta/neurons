function [I1] = rect_vis_ind45(I,r,c,pol, varargin)
% visualize haar-like rectangles
%
% rect_vis_ind45(I,r,c,pol, D)
%     I = zeros, size of window, include padding [25 26]
%     r = cell containing rectangles linear indexes
%     c = matrix containing rectangle colors
%     pol = polarity of the feature (switches the colors)
%     D = display (1 = yes, 0 = no)

% toggle the display (D = 1
if nargin > 4
    D = varargin{1};
else
    D = 1;
end
if nargin < 4
    pol = 1;
end

%VEC_SHIFT = (size(I,1)+1)  * (size(I,2)+1);
VEC_SHIFT = 25*25;
%IISIZE = size(I) + [ 2 1];
IISIZE = size(I);

N = length(r);

BW0 = zeros(IISIZE); BW0 = boolean(BW0); I1 = zeros(size(BW0));

for n = 1:N
    
    rect = r{n};
    rect = rect - VEC_SHIFT;
    
    [R, C] = ind2sub(IISIZE, rect);
    col = c(n);
    
    BW = BW0;
    
    % draw a line from A to C+[1 1]
    BW = drawline(BW, [R(1) C(1)], [R(3)+1 C(3)+1], [-1 -1]);
    
    % draw a line from C+[0 1] to D+[1 0]
    BW = drawline(BW, [R(3) C(3)+1], [R(4)+1 C(4)], [-1 1]);
    
    % draw a line from D+[1 0] to B+[0 -1]
    BW = drawline(BW, [R(4)+1 C(4)], [R(2) C(2)-1], [1 1]);

    % draw a line from A to B+[1 -1]
    BW = drawline(BW, [R(1) C(1)], [R(2)+1 C(2)-1], [-1 1]);
    
    %keyboard;
    BW = imfill(BW,[R(1)-1 C(1)],4);  % fill the center
    
    
    
    
    I1 = I1 + col*BW;
    
%     I1(R(1),C(1)) = 2;
%     I1(R(2),C(2)) = 2;
%     I1(R(3),C(3)) = 2;
%     I1(R(4),C(4)) = 2;
    
    %keyboard;
end

% switch to normal [24 24] IMSIZE for display
I1 = I1(2:size(I1,1), 2:size(I1,2)-1);


if pol == -1
    I1 = pol*I1;
end

if D
    imagesc(I1, [-1 1]); colormap gray;
    drawnow;
    pause(0.005);
end




function BW = drawline(BW,start,stop,vec)

r = start(1); c = start(2);
pr = sign(stop(1) - start(1));  % + indicated increasing from start to stop
pc = sign(stop(2) - start(2));  % + indicates increasing from start to stop

if start(1) == stop(1)
    BW(r,c) = 1;
else

    while (pr*r <= pr*stop(1)) && (r > 0) && (r <= size(BW,1)) && (pc*c <= pc*stop(2)) && (c > 0) && (c <= size(BW,2))
        BW(r,c) = 1;
        r = r+vec(1);
        c = c+vec(2);
    end
end

%keyboard;


