function  FEATURE = HoG(I, varargin)
%
% F = hog(I);
% F = hog(I, 'orientationbins', 'cellsize', 'blocksize')
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

%keyboard;

%% the final feature is the CELLS normalized with 4 possible neighbor combos
FEATURE = zeros([size(CELLS,1) size(CELLS,2) size(CELLS,3) 4]);
neighbors = 1:4;


%% normalize the histograms in each block according to its neighbors
for r=1:blocksize(1):size(CELLS,1)+1
    for c = 1:blocksize(2):size(CELLS,2)+1
        for n = neighbors;
            
            % find the appropriate rows to normalize
            switch n
                case 1      % neighbors are up & left
                    rows = max(1,r - blocksize(1)+1):min(r,size(CELLS,1));
                    cols = max(1,c - blocksize(2)+1):min(c, size(CELLS,2));
                    
                case 2      % neighbors are up & right
                    rows = max(1,r - blocksize(1)+1):min(r,size(CELLS,1));
                    cols = c: min(c+ blocksize(2)-1, size(CELLS,2));
                    
                case 3      % neighbors are down & right
                    rows = r: min(r+ blocksize(1)-1, size(CELLS,1));
                    cols = c: min(c+ blocksize(2)-1, size(CELLS,2));
                    
                case 4      % neighbors are down & left
                    rows = r: min(r+ blocksize(1)-1, size(CELLS,1));
                    cols = max(1,c - blocksize(2)+1):min(c, size(CELLS,2));
            end
%             rows = r: r+ blocksize(1)-1;
%             cols = c: c+ blocksize(2)-1;
%             disp(['case ' num2str(n) ' block rows: ' num2str(rows) '  cols: ' num2str(cols) '  size = [' num2str(size(CELLS(rows,cols,:))) ']']);
        
%             disp(['size = [' num2str(size(CELLS(rows,cols,:))) ']' ]);

            FEATURE(rows,cols,:,n) = l2hys(CELLS(rows,cols,:));
        end
    end
end


% % normalize the histograms in each block
% for r=1:blocksize(1):size(CELLS,1)
%     for c = 1:blocksize(2):size(CELLS,2)
%         rows = r: r+ blocksize(1)-1;
%         cols = c: c+ blocksize(2)-1;
%         %disp(['block rows: ' num2str(rows) '   cols: ' num2str(cols)]);
%         
%         CELLS(rows,cols,:) = l2hys(CELLS(rows,cols,:));
%         
%     end
% end





%% Visualize!
% ANGL = rad2deg(ANGL);
% %plot the gradient
% for x=1:size(I,1)
%     for y=1:size(I,2)
%         X(x,y) = x;  Y(x,y) = y;
%     end
% end
% imshow(I); hold on;  axis image; set(gca, 'Position', [0 0 1 1]);
% quiver(Y(:),X(:), GradX(:), GradY(:));
% 
% figure;
%hogview(I);




%% supporting functions

function CELLBIN = sumsum(c)

CELLBIN = repmat(sum(sum(c)), size(c));







