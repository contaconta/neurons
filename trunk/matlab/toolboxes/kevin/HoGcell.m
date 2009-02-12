function  F = HoGcell(I, varargin)
%
% F = hogcell(I,cellr,cellc,bin);
% F = hogcell(I,ellr,cellc,bin, 'orientationbins', 'cellsize', 'blocksize', 'decompress')
%
%

orientationbins = 9;  cellsize = [4 4];  blocksize = [2 2]; DECOMPRESS = 0;

if nargin > 1
    for i = 1:nargin-1
        if strcmp('orientationbins', varargin{i})
            orientationbins = varargin{i+1};
        end
        if strcmp('cellsize', varargin{i})
            cellsize = varargin{i+1};
        end
        if strcmp('blocksize', varargin{i})
            blocksize = varargin{i+1};
        end
        if strcmp('decompress', varargin{i})
            DECOMPRESS = 1;
        end
        if strcmp('CELLS', varargin{i})
            CELLS = 1;
        end
    end
end


% first, apply the gradient in x and y
filtX = [1 0 -1];
filtY = filtX';

GradX = imfilter(I,filtX, 'symmetric');
GradY = imfilter(I,filtY, 'symmetric');


% compute the gradient magnitude, and gradient orientation
NORM = arrayfun(@(a,b)(norm([a b])), GradX, GradY);
ANGL = arrayfun(@(a,b)(atan2(a,b)), GradY, GradX);

% normalize the orientations so they fall beteen 0-180 degrees
ANGL = ANGL + (pi)*(ANGL < 0);

% define the centers of each orientation bin
bins = 0: pi/(orientationbins-1) :pi;  bin_width = bins(2);

% quantize the angles to the nearest bin
ANGL = quant(ANGL,bin_width);
ANGLINDS = (ANGL *((length(bins)-1)/pi) + 1);

% build a histogram for each cell
for i=1:length(bins)
    NORMS(:,:,i) = (ANGLINDS == i) .* NORM;
end

for i=1:length(bins)
    HIST(:,:,i) = blkproc(NORMS(:,:,i), cellsize, @sumsum);
end

CELLS = HIST(1:cellsize(1):size(I,1), 1:cellsize(2):size(I,2), :);

% normalize the histograms in each block
for r=1:blocksize(1):size(CELLS,1)
    for c = 1:blocksize(2):size(CELLS,2)
        rows = r: r+ blocksize(1)-1;
        cols = c: c+ blocksize(2)-1;
        %disp(['block rows: ' num2str(rows) '   cols: ' num2str(cols)]);
        
        CELLS(rows,cols,:) = l2hys(CELLS(rows,cols,:));
        
    end
end

if DECOMPRESS
    %F = CELLS;
else
    F = CELLS; 
end



%ANGL = rad2deg(ANGL);
% %plot the gradient
% for x=1:size(I,1)
%     for y=1:size(I,2)
%         X(x,y) = x;  Y(x,y) = y;
%     end
% end
% imshow(I); hold on;  axis image; set(gca, 'Position', [0 0 1 1]);
% quiver(Y(:),X(:), GradX(:), GradY(:));


%% supporting functions

function CELLBIN = sumsum(c)

CELLBIN = repmat(sum(sum(c)), size(c));

