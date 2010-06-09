function blankI = rect_vis_ind(blankI,rect,col,pol)
% visualize haar-like rectangles
%
% rect_vis_ind(blankI,rect,col,pol)
%     blankI = zeros, size of window
%     rect = cell containing rectangles linear indexes
%     col = matrix containing rectangle colors
%     pol = polarity of the feature (switches the colors)


if ~exist('pol', 'var')
    pol = 1;
end

%figure(3436321);

N = length(rect);

if rect{1}(1) > (size(blankI,1)+1) * (size(blankI,2)+1);
    rect_vis_ind45(zeros(size(blankI)+[1 2]), rect, col,1);
else
    for n = 1:N  
        [rectR, rectC] = ind2sub(size(blankI)+[1 1], rect{n});
        col1 = col(n);
        blankI(rectR(1):rectR(4)-1,rectC(1):rectC(4)-1) = blankI(rectR(1):rectR(4)-1,rectC(1):rectC(4)-1) + pol*col1;
    end

    imagesc(blankI, [-1 1]); colormap gray;

    drawnow;
    pause(0.005);
end