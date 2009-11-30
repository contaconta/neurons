%% PATH INFO
EXPNAME = 'shapeContextRmax400T12';

addpath('/osshare/Work/neurons/matlab/toolboxes/shapecontext_demo/');
addpath('/osshare/Work/neurons/matlab/features/spedges/');
imgpath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
superpixelpath = '/osshare/DropBox/Dropbox/aurelien/superpixels/labels/';
annotationpath = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
localresultpath = ['./featurevectors/' EXPNAME '/'];
if ~isdir(localresultpath);mkdir(localresultpath);end

d = dir([annotationpath '*.png']);

%% PARAMETERS
Rbins = 5;
Rmin = 8;
Rmax = 400;
Tbins = 12;

DEPEND = [1 1; 2 2; 3 62];


for f = 1:length(d)
    %% get labels, annotation, superpixels, image, ...
    clear RAY1 RAY2 RAY3 RAY4 A L G STATS superpixels RAYFEATUREVECTOR;
    FILEROOT = regexp(d(f).name, '(\w*)[^\.]', 'match');
    FILEROOT = FILEROOT{1};
    disp(['reading ' FILEROOT]);
    I = imread([imgpath FILEROOT '.png']);
    A = imread([annotationpath FILEROOT '.png']); A = A(:,:,2) > 200;
    disp('getting the superpixel labels');
    L = readRKLabel([superpixelpath FILEROOT '.dat'], [size(I,1) size(I,2)]);
 	L = L';
    superpixels = unique(L(:))';
    STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
    labels = zeros(size(superpixels));
    
    for l = superpixels
        STATS(l).Centroid = round(STATS(l).Centroid);
        if sum(A(STATS(l).PixelIdxList)) >= 0.5* STATS(l).Area
            mito(l) = 1;
        else
            mito(l) = 0;
        end
    end
    
%    	gh = imfilter(I,fspecial('sobel')' /8,'replicate');
%     gv = imfilter(I,fspecial('sobel')/8,'replicate');
%     G(:,:,1) = gv;
%     G(:,:,2) = gh;  clear gh gv;
    
    %% fill the featureVector
    RAYFEATUREVECTOR = zeros([length(superpixels) (Rbins*Tbins)+2]);

    disp('computing intensity cues!');
    for l = superpixels
        RAYFEATUREVECTOR(l, 1) = mean(I(STATS(l).PixelIdxList));
        RAYFEATUREVECTOR(l, 2) = var(double(I(STATS(l).PixelIdxList)));
    end

    %% do shape context stuff here!
    disp('computing the shape context for each centroid!');
    EDGE = niceEdge(I);  if f == 1; imwrite(imoverlay(I, EDGE), [localresultpath FILEROOT '.png'], 'PNG'); end;     
    [Er Ec] = find(EDGE == 1);
    Es = [Er Ec];
    
    tic;
    for l = superpixels
        if mod(l, 500) == 0
            disp(['  (' num2str(l) '/' num2str(length(superpixels)) ')']);
            toc;
        end
        centroid = [STATS(l).Centroid(2) STATS(l).Centroid(1)];
        %MASK = zeros(size(I));
        %MASK(max(1,centroid(1)-Rmax):min(centroid(1)+Rmax, size(I,1)), max(1,centroid(2)-Rmax):min(centroid(2)+Rmax, size(I,2))) = 1;
        %EDGE1 = EDGE.*MASK;
        %[Er Ec] = find(EDGE1 == 1);
        
        E1 = Es( (Es(:,1) >= centroid(1)-Rmax) & (Es(:,1) <= centroid(1)+Rmax), :);
        E1 = E1( (E1(:,2) >= centroid(2)-Rmax) & (E1(:,2) <= centroid(2)+Rmax), :);
        
        %S = shape_context(centroid, [Er Ec], Rmin, Rmax, Rbins, Tbins);
        S = shape_context(centroid, E1, Rmin, Rmax, Rbins, Tbins);
        RAYFEATUREVECTOR(l, 3:62) = S; 
        %imagesc(S); drawnow; pause(0.02);
    end
    
    save([localresultpath FILEROOT '.mat'], 'RAYFEATUREVECTOR', 'L', 'superpixels', 'mito', 'DEPEND');

end
