function rect_vis(blankI,r,c,p)
% visualize haar-like rectangles
%
% rect_vis(blankI,r,c,p)
%     blankI = zeros, size of window
%     r = cell containing rectangles rows
%     c = cell containing rectangles columns
%     p = matrix containing rectangles polarities
%



%figure(975675);


N = length(r);

for n = 1:N

    rectR = r{n};
    rectC = c{n};
    pol = p(n);

    
    blankI(rectR(1):rectR(4)-1,rectC(1):rectC(4)-1) = pol;
end

imagesc(blankI); colormap gray;

drawnow;
pause(0.005);
