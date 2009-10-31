 

A = imread(['/osshare/DropBox/Dropbox/aurelien/shapeFeatureVectors/zero_or_one/' 'FIBSLICE1627_2ANN.png']); 
A = A(:,:) > 200;


%L = readRKLabel([superpixelpath FILEROOT '.dat'], [size(I,1) size(I,2)]);
L = readRKLabel([superpixelpath 'FIBSLICE1627_2.dat'], [size(I,1) size(I,2)]);
L = L';
superpixels = unique(L(:))';
STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
mito = zeros(size(superpixels));


for l = superpixels
    STATS(l).Centroid = round(STATS(l).Centroid);
    if sum(A(STATS(l).PixelIdxList)) >= 0.5* STATS(l).Area
        mito(l) = 1;
    end
end



fid = fopen(['/osshare/DropBox/Dropbox/aurelien/shapeFeatureVectors/zero_or_one/' 'FIBSLICD1627_2_all_feature_vectors'], 'a');



for s = superpixels
    
    
    if mito(s) == 1
        fprintf(fid, '%d ', 2);
    

        fprintf(fid, '%d:%g\n', 1, 2);
    else
        fprintf(fid, '%d ', 0);

        fprintf(fid, '%d:%g\n', 1, 0);
    end
    
end

fclose(fid);