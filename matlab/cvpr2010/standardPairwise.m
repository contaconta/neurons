resultname = 'RayHistSteer';

raysFolderName = 'rays30MedianInvariantE2';

histFolder = '/osshare/DropBox/Dropbox/aurelien/FeatureVectors/histogram/';
featureFolder = ['./featurevectors/' raysFolderName '/'];
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
boundaryFolder = '/osshare/DropBox/Dropbox/aurelien/superpixels/annotations/';
imgFolder = '/osshare/Work/Data/LabelMe/Images/fibsem/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/pairwise/' resultname '/'];
adjacencyFolder =  '/osshare/DropBox/Dropbox/aurelien/superpixels/neighbors/';
steerableFolder = './featurevectors/steerable_featureVectors/';
if ~isdir(destinationFolder); mkdir(destinationFolder); end;

pairwiseFolder = '/osshare/DropBox/Dropbox/aurelien/pairwise/RayHistSteer/';


addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');

D = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
for x = 105:204
    D(size(D,1)+1,:) = [x x]; % Intensity & steer
end
D2 = D + 204;
D = [D; D2];
DMAX = 408;
BOUNDARY_LABEL = 1;


fileRoot = 'FIBSLICE0880';

% load the SVM model
load([pairwiseFolder 'svm_model11  12  13  14  15.mat' ]);

%P = libsvmread([pairwiseFolder fileRoot '.txt']);

% load the features
load([featureFolder fileRoot '.mat']); 
[lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
H = full(H);
[lab S] = libsvmread([steerableFolder fileRoot '_u0_all_feature_vectors']);
S = full(S);

C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
load([adjacencyFolder fileRoot '.mat']);  

% bg, boundary, mito labels needed for labeling boundaries pairs
labels = zeros(size(STATS));
for l=1:length(STATS)
    labels(l) = mode( C(STATS(l).PixelIdxList) );
end

% make an adjacency matrix of edges we need to evaluate
E = triu(A) - speye(size(A));
[r c] = find(E ~= 0);


% determine centroid locations on adjacency graph
locs = zeros(length(superpixels), 2);
for s = superpixels
    locs(s,:) = STATS(s).Centroid;
end


DIFFS = zeros(size(r));
for x = 1:length(r)
   DIFFS(x) =  RAYFEATUREVECTOR(r(x),1) - RAYFEATUREVECTOR(c(x),1);    
end

% compute the histogram pairwise
SIGMA = mean(DIFFS);
SIGMA2sq = 2*(SIGMA^2);


phi = zeros(size(r));
for x = 1:length(r)
    phi(x) = norm([RAYFEATUREVECTOR(r(x),1) -  RAYFEATUREVECTOR(c(x),1)])^2 / SIGMA2sq;
end
phi = phi/max(phi);
phi = exp(phi);
phi = phi/max(phi);

P = sparse([],[],[], size(A,1), size(A,2),0);
for x = 1:length(r)
    P(r(x),c(x)) = phi(x);
end
P = max(P,P');

hold off; figure(2); cla; imshow(I); hold on;
colors = jet(25); %17:24

gplot2(P > .5 ,locs, '.-', 'Color', colors(17,:));
gplot2(P > .5714 ,locs, '.-', 'Color', colors(18,:));
gplot2(P > .6429 ,locs, '.-', 'Color', colors(19,:));
gplot2(P > .7143 ,locs, '.-', 'Color', colors(20,:));
gplot2(P > .7857 ,locs, '.-', 'Color', colors(21,:));
gplot2(P > .8571 ,locs, '.-', 'Color', colors(22,:));
gplot2(P > .9286 ,locs, '.-', 'Color', colors(23,:));
gplot2(P > 1 ,locs, '.-', 'Color', colors(24,:));
