function rect_vis_ind(blankI,i,p)
% visualize haar-like rectangles
%
% rect_vis_ind(blankI,i,p)
%     blankI = zeros, size of window
%     r = cell containing rectangles linear indexes
%     p = matrix containing rectangles polarities
%



%figure(3436321);


N = length(i);

for n = 1:N
    
    [rectR, rectC] = ind2sub(size(blankI)+[1 1], i{n});
    %[rectC, rectR] = ind2sub(size(blankI)+[1 1], i{n});
    %rectR = r{n};
    %rectC = c{n};
    pol = p(n);

    
    blankI(rectR(1):rectR(4)-1,rectC(1):rectC(4)-1) = pol;
end

imagesc(blankI, [-1 1]); colormap gray;

drawnow;
pause(0.005);