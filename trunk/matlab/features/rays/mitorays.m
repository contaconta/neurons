%% PATH INFO
EXPNAME = 'simpleTest2';

addpath('/osshare/Work/neurons/matlab/features/spedges/');
imgpath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
superpixelpath = '/osshare/DropBox/Dropbox/aurelien/superpixels/labels/';
annotationpath = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
dropboxresultpath = ['/osshare/DropBox/Dropbox/aurelien/shapeFeatureVectors/' EXPNAME '/'];
localresultpath = ['./featurevectors/' EXPNAME '/'];
if ~isdir(dropboxresultpath);mkdir(dropboxresultpath);end
if ~isdir(localresultpath);mkdir(localresultpath);end

d = dir([annotationpath '*.png']);
libsvmFileName = 'feature_vectors';

%% PARAMETERS
angles = 0:20:340;
combos = combnk(angles, 2);
stride = 1;
eta = 1;




for f = 1:length(d)
    clear RAY1 RAY2 RAY3 RAY4;
    
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
    mito = zeros(size(superpixels));
    
    gh = imfilter(I,fspecial('sobel')' /8,'replicate');
    gv = imfilter(I,fspecial('sobel')/8,'replicate');
    G(:,:,1) = gv;
    G(:,:,2) = gh;
    
    RAYFEATUREVECTOR = zeros([length(superpixels) 3*length(angles) + size(combos,1) + 1]);
    
    for l = superpixels
        STATS(l).Centroid = round(STATS(l).Centroid);
        if sum(A(STATS(l).PixelIdxList)) >= 0.5* STATS(l).Area
            mito(l) = 1;
        end
    end
    
    
    


    % RAY 1 is the basic ray, the distance RAY
    RAY1 = zeros([size(I) length(angles)]);
    RAY3 = zeros([size(I) length(angles)]);
    RAY4 = zeros([size(I) length(angles)]);

    EDGE = niceEdge(I);  if f == 1; imwrite(imoverlay(I, EDGE), [localresultpath FILEROOT '.png'], 'PNG'); imwrite(imoverlay(I, EDGE), [dropboxresultpath FILEROOT '.png'], 'PNG');end; 

    for l = superpixels
        RAYFEATUREVECTOR(l, 1) = mean(I(STATS(l).PixelIdxList));
    end
    
    for i = 1:length(angles)
        disp(['computing R1 R3 R4 for angle = ' num2str(angles(i))]);
        %R = spedge_dist(I, angles(i), stride, EDGE);  
        [R1 R3 R4] = rays(EDGE, G, angles(i), stride);
        RAY1(:,:,i) = R1;
        RAY3(:,:,i) = R3;
        RAY4(:,:,i) = R4;
        %[RAY3(:,:,i) RAY4(:,:,i)] = spangle_dist(I,angles(i), stride, EDGE);
        
        for l = superpixels
            % store the median ray in the superpixel
            %RAYFEATUREVECTOR(l, i+1) =  median(R1(STATS(l).PixelIdxList));
            %RAYFEATUREVECTOR(l, length(angles) + i+1) = median(R3(STATS(l).PixelIdxList));
            %RAYFEATUREVECTOR(l, 2*length(angles) + i+1) = median(R4(STATS(l).PixelIdxList));
            % store the centroid ray in the superpixel
            RAYFEATUREVECTOR(l,i) = R1(STATS(l).Centroid(2), STATS(l).Centroid(1));
            RAYFEATUREVECTOR(l, i+1) =  R1(STATS(l).Centroid(2), STATS(l).Centroid(1));
            RAYFEATUREVECTOR(l, length(angles) + i+1) = median(STATS(l).Centroid(2), STATS(l).Centroid(1));
            RAYFEATUREVECTOR(l, 2*length(angles) + i+1) = median(STATS(l).Centroid(2), STATS(l).Centroid(1));
        end
            
    end


    
   
    
    % RAY2
    disp('computing difference ray feature');
    pause(0.001);
    for c = 1:size(combos,1);
        disp([' raydiff ' num2str(combos(c,1)) ' ' num2str(combos(c,2))]);
        angle1 = angles == combos(c,1);
        angle2 = angles == combos(c,2);
        RAY2 = (RAY1(:,:,angle1) - RAY1(:,:,angle2)) ./ (RAY1(:,:,angle1)+eta);
        
        % MAKE THE DIFFERENCE RAY MORE PICKY
        %RAY2 = exp(RAY2);
        
        
        for l = superpixels           
            % store the median ray in the superpixel
            %RAYFEATUREVECTOR(l, 3*length(angles) + c) = median(RAY2(STATS(l).PixelIdxList));
            % store the centroid ray in the superpixel
            RAYFEATUREVECTOR(l, 3*length(angles) + c+1) = RAY2(STATS(l).Centroid(2), STATS(l).Centroid(1));
        end
    end

    save([localresultpath FILEROOT '.mat'], 'RAYFEATUREVECTOR', 'L', 'superpixels', 'mito');
    
    %% Write to a LIBSVM file a random sampling of the feature vector
%     N = 200;
%     writeLIBSVMfeaturevector(RAYFEATUREVECTOR, L, superpixels, mito,libsvmFileName, dropboxresultpath, N);
end


   