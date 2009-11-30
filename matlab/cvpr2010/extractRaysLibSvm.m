raysName = 'rays30MedianInvariantE2';

featureFolder = ['./featurevectors/' raysName '/'];
boundaryFolder = '/osshare/DropBox/Dropbox/aurelien/superpixels/annotations/';
destinationFolder = './featurevectors/extractedRaysHistNorm/';
if ~isdir(destinationFolder); mkdir(destinationFolder); end;

addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');



%libsvmwrite('SPECTFlibsvm.train', labels, features_sparse);

load D2.mat;

d = dir([featureFolder '*.mat']);

for i = 1:length(d)

    disp(['loading ' d(i).name ]);
    
    load([featureFolder d(i).name]); 
    [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
    H = full(H);
    
    fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
    fileRoot = fileRoot{1};
    
    C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        
    STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
    clear labels; labels = zeros(size(RAYFEATUREVECTOR,1),1);
    for l=1:length(STATS)
        labels(l) = mode(C(STATS(l).PixelIdxList) ); 
    end
    
    featureVector = [RAYFEATUREVECTOR H];
    
    % rescale the data
    for x = 1:size(D,1)
        featureVector(:,D(x,1):D(x,2)) = mat2gray(featureVector(:,D(x,1):D(x,2)), limits(x,:));
    end
    
    featureVector = sparse(featureVector);
    
    disp(['writing ' featureFolder fileRoot '_u0_all_feature_vectors']);
    
    libsvmwrite([destinationFolder fileRoot '_u0_all_feature_vectors'], labels, featureVector);
    
    
    
    
end