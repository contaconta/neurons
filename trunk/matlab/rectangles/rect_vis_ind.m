function blankI = rect_vis_ind(blankI,i,c,pol)
% visualize haar-like rectangles
%
% rect_vis_ind(blankI,i,p)
%     blankI = zeros, size of window
%     r = cell containing rectangles linear indexes
%     p = matrix containing rectangles polarities
%


if ~exist('pol', 'var')
    pol = 1;
end

%figure(3436321);


N = length(i);

for n = 1:N
    
    [rectR, rectC] = ind2sub(size(blankI)+[1 1], i{n});
    %[rectC, rectR] = ind2sub(size(blankI)+[1 1], i{n});
    %rectR = r{n};
    %rectC = c{n};
    col = c(n);

    
    blankI(rectR(1):rectR(4)-1,rectC(1):rectC(4)-1) = pol*col;
end

imagesc(blankI, [-1 1]); colormap gray;

drawnow;
pause(0.005);