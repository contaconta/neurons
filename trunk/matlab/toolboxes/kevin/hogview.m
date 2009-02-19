function Feature = hogview(I, varargin)

cellsize = [4 4]; blocksize = [2 2]; orientationbins = 9; neighborcase = 1;

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
        if strcmp('neighborcase', varargin{i})
            neighborcase = varargin{i+1};
        end
    end
end

Feature = Hog(I, 'cellsize', cellsize, 'blocksize', blocksize, 'orientationbins',orientationbins );
F = Feature(:,:,:,neighborcase);


imshow(I); set(gca, 'Position', [0 0 1 1]); hold on;

switch neighborcase
    case 1
        % horizontal block lines
        horzY1 = (blocksize(1)*cellsize(1))/2:cellsize(1)*blocksize(1):size(I,1);
        horzY2 = horzY1;
        horzX1 = zeros(size(horzY1));
        horzX2 = size(I,2)*ones(size(horzY1));
        horzY1 = horzY1 + .5; horzY2 = horzY2+.5; horzX1 = horzX1+.5; horzX2=horzX2+.5;
        
        % vertical block lines
        vertX1 = (blocksize(2)*cellsize(2))/2:cellsize(2)*blocksize(2):size(I,2);
        vertX2 = vertX1;
        vertY1 = zeros(size(vertX1));
        vertY2 = size(I,1)*ones(size(vertX1)); 
        vertY1 = vertY1 + .5; vertY2 = vertY2+.5; vertX1 = vertX1+.5; vertX2=vertX2+.5;
    case 2
        % horizontal block lines
        horzY1 = (blocksize(1)*cellsize(1))/2:cellsize(1)*blocksize(1):size(I,1);
        horzY2 = horzY1;
        horzX1 = zeros(size(horzY1));
        horzX2 = size(I,2)*ones(size(horzY1));
        horzY1 = horzY1 + .5; horzY2 = horzY2+.5; horzX1 = horzX1+.5; horzX2=horzX2+.5;
        
        % vertical block lines
        vertX1 = 0:cellsize(2)*blocksize(2):size(I,2);
        vertX2 = vertX1;
        vertY1 = zeros(size(vertX1));
        vertY2 = size(I,1)*ones(size(vertX1)); 
        vertY1 = vertY1 + .5; vertY2 = vertY2+.5; vertX1 = vertX1+.5; vertX2=vertX2+.5;
        
    case 3  % down and right
        % horizontal block lines
        horzY1 = 0:cellsize(1)*blocksize(1):size(I,1);
        horzY2 = horzY1;
        horzX1 = zeros(size(horzY1));
        horzX2 = size(I,2)*ones(size(horzY1));
        horzY1 = horzY1 + .5; horzY2 = horzY2+.5; horzX1 = horzX1+.5; horzX2=horzX2+.5;

        % vertical block lines
        vertX1 = 0:cellsize(2)*blocksize(2):size(I,2);
        vertX2 = vertX1;
        vertY1 = zeros(size(vertX1));
        vertY2 = size(I,1)*ones(size(vertX1)); 
        vertY1 = vertY1 + .5; vertY2 = vertY2+.5; vertX1 = vertX1+.5; vertX2=vertX2+.5;
        
    case 4
        % horizontal block lines
        horzY1 = 0:cellsize(1)*blocksize(1):size(I,1);
        horzY2 = horzY1;
        horzX1 = zeros(size(horzY1));
        horzX2 = size(I,2)*ones(size(horzY1));
        horzY1 = horzY1 + .5; horzY2 = horzY2+.5; horzX1 = horzX1+.5; horzX2=horzX2+.5;
        
        % vertical block lines
        vertX1 = (blocksize(2)*cellsize(2))/2:cellsize(2)*blocksize(2):size(I,2);
        vertX2 = vertX1;
        vertY1 = zeros(size(vertX1));
        vertY2 = size(I,1)*ones(size(vertX1)); 
        vertY1 = vertY1 + .5; vertY2 = vertY2+.5; vertX1 = vertX1+.5; vertX2=vertX2+.5;
end



for i=1:length(horzY1)
    line([horzX1(i) horzX2(i)], [horzY1(i) horzY2(i)]);
end
for i=1:length(vertY1)
    line([vertX1(i) vertX2(i)], [vertY1(i) vertY2(i)]);
end

% orientations
orientations = 0: pi/(size(F,3)-1) :pi;


% plot the normalized histograms in each cell
for r = 1:size(F,1)
    for c = 1:size(F,2)
        x = (c-1)*cellsize(1)+(cellsize(1))/2 +.5;
        y = (r-1)*cellsize(2)+(cellsize(2))/2 +.5;
        plot(x,y, 'b.');
        
        for j = 1:size(F,3)
            qx(j) = x;
            qy(j) = y;
            qu(j) = 1.5*cellsize(2)*F(r,c,j)*cos(orientations(j));
            qv(j) = 1.5*cellsize(1)*F(r,c,j)*sin(orientations(j));
        end
        quiver(qx,qy,qu,qv);
    end
end


% vertX =
% vertY =


%ANGL = rad2deg(ANGL);
% %plot the gradient
% for x=1:size(I,1)
%     for y=1:size(I,2)
%         X(x,y) = x;  Y(x,y) = y;
%     end
% end
% imshow(I); hold on;  axis image; set(gca, 'Position', [0 0 1 1]);
% quiver(Y(:),X(:), GradX(:), GradY(:));