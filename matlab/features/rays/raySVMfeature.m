function raySVMfeature(I, L, A, libsvmFileName, resultpath)

%% PATH INFO


addpath('/osshare/Work/neurons/matlab/features/spedges/');
%imgpath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
%superpixelpath = '/osshare/DropBox/Dropbox/aurelien/superpixels/labels/';
%annotationpath = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
%resultpath = ['/osshare/DropBox/Dropbox/aurelien/shapeFeatureVectors/' EXPNAME '/'];
if ~isdir(resultpath)
    mkdir(resultpath);
end


%% PARAMETERS
angles = 0:30:330;
combos = combnk(angles, 2);
stride = 1;
eta = 1;




for f = 1:1
%     FILEROOT = regexp(d(f).name, '(\w*)[^\.]', 'match');
%     FILEROOT = FILEROOT{1};
%     disp(['reading ' FILEROOT]);
%     I = imread([imgpath FILEROOT '.png']);
%     A = imread([annotationpath FILEROOT '.png']); A = A(:,:,2) > 200;
%     disp('getting the superpixel labels');
%     L = readRKLabel([superpixelpath FILEROOT '.dat'], [size(I,1) size(I,2)]);
%  	L = L';
    superpixels = unique(L(:))';
    STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
    
    gh = imfilter(I,fspecial('sobel')' /8,'replicate');
    gv = imfilter(I,fspecial('sobel')/8,'replicate');
    G(:,:,1) = gv;
    G(:,:,2) = gh;
    
    for l = superpixels
        STATS(l).Centroid = round(STATS(l).Centroid);
        if sum(A(STATS(l).PixelIdxList)) >= 0.5* STATS(l).Area
            mito(l) = 1;
        else
            mito(l) = 0;
        end
    end
    
    
    RAYFEATUREVECTOR = zeros([length(superpixels) length(angles) + size(combos,1)]);
    %RAYFEATUREVECTOR = zeros([length(superpixels) 3*length(angles) + size(combos,1)]);


    % RAY 1 is the basic ray, the distance RAY
    RAY1 = zeros([size(I) length(angles)]);
    RAY3 = zeros([size(I) length(angles)]);
    RAY4 = zeros([size(I) length(angles)]);

    EDGE = niceEdge(I);  imwrite(imoverlay(I, EDGE), [resultpath 'edgesused.png'], 'PNG'); 

    
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
            RAYFEATUREVECTOR(l, i) =  median(R1(STATS(l).PixelIdxList));
            RAYFEATUREVECTOR(l, length(angles) + i) = median(R3(STATS(l).PixelIdxList));
            RAYFEATUREVECTOR(l, 2*length(angles) + i) = median(R4(STATS(l).PixelIdxList));
            % store the centroid ray in the superpixel
            %RAYFEATUREVECTOR(l,i) = R1(STATS(l).Centroid(2), STATS(l).Centroid(1));
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
        RAY2 = exp(RAY2);
        
        
        for l = superpixels           
            % store the median ray in the superpixel
            %RAYFEATUREVECTOR(l, length(angles) + c) = median(RAY2(STATS(l).PixelIdxList));
            RAYFEATUREVECTOR(l, 3*length(angles) + c) = median(RAY2(STATS(l).PixelIdxList));
            % store the centroid ray in the superpixel
            %RAYFEATUREVECTOR(l,length(angles) + c) = RAY2(STATS(l).Centroid(2), STATS(l).Centroid(1));
        end
    end

    save([resultpath libsvmFileName 'FeatureVector.mat'], 'RAYFEATUREVECTOR', 'L', 'superpixels');
    
    %% Write to a LIBSVM file a random sampling of the feature vector
    
    writeLIBSVMclassificationVector(RAYFEATUREVECTOR, superpixels, libsvmFileName, resultpath, mito);
%     writeLIBSVMfeaturevector(RAYFEATUREVECTOR, L, superpixels, mito,libsvmFileName, resultpath, N);
end

keyboard;