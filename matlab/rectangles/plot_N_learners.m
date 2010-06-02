function plot_N_learners(rects, cols, pol, IMSIZE, N1)

if ~exist('N1', 'var')
    N = length(rects);
    Nsqrt = floor(sqrt(N));
    N1 = Nsqrt*Nsqrt;
end

%T = length(CLASSIFIER.rects);

blankI = zeros(IMSIZE);



for i = 1:N1
    if i <= length(rects)
        rect = rects{i};
        col = cols{i};
        p = pol(i);

        subplottight(Nsqrt,Nsqrt,i);
        rect_vis_ind(blankI,rect,col,p);
        axis off; axis image;
        h = line([0.5 IMSIZE(2)+.5 IMSIZE(2)+.5 0.5 0.5], [0.5 0.5 IMSIZE(1)+.5 IMSIZE(1)+.5 0.5]);
        set(h, 'Color', [0 0 0]);
    end
end


function h=subplottight(Ny, Nx, j, margin)
% General utility function
%
% This function is like subplot but it removes the spacing between axes.
%
% subplottight(Ny, Nx, j)

if nargin <4 
    margin = 0;
end

j = j-1;
x = mod(j,Nx)/Nx;
y = (Ny-fix(j/Nx)-1)/Ny;
h=axes('position', [x+margin/Nx y+margin/Ny 1/Nx-2*margin/Nx 1/Ny-2*margin/Ny]);
