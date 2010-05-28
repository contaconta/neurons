function plot_100_learners(CLASSIFIER, IMSIZE)


T = length(CLASSIFIER.rects);

rects = CLASSIFIER.rects(1:T);
thresh = CLASSIFIER.thresh(1:T);
alpha = CLASSIFIER.alpha(1:T);

% backwards-compatible naming
if isfield(CLASSIFIER, 'pols')
    cols = CLASSIFIER.pols(1:T);
else
    cols  = CLASSIFIER.cols(1:T);
end
if isfield(CLASSIFIER, 'tpol')
    pol = CLASSIFIER.tpol(1:T);
else
    pol = CLASSIFIER.pol(1:T);
end

blankI = zeros(IMSIZE);

for i = 1:100
    rect = rects{i};
    col = cols{i};
    p = pol(i);

    subplottight(10,10,i);
    rect_vis_ind(blankI,rect,col,p);
    axis off; axis image;
    h = line([0.5 IMSIZE(2)+.5 IMSIZE(2)+.5 0.5 0.5], [0.5 0.5 IMSIZE(1)+.5 IMSIZE(1)+.5 0.5]);
    set(h, 'Color', [0 0 0]);
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
