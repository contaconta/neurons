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


fileRoot = 'FIBSLICE1560';

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

% we need 2 feature vectors because we must check both directions of the pairwise term
featureVector1 = zeros(length(r), DMAX);
featureVector2 = zeros(length(r), DMAX);
L1 = zeros(length(r),1);
L2 = zeros(length(r),1);

for x = 1:length(r)
    featureVector1(x,:) = [RAYFEATUREVECTOR(r(x),:) H(r(x),:) S(r(x),:) RAYFEATUREVECTOR(c(x),:) H(c(x),:) S(c(x),:)];
    featureVector2(x,:) = [RAYFEATUREVECTOR(c(x),:) H(c(x),:) S(c(x),:) RAYFEATUREVECTOR(r(x),:) H(r(x),:) S(r(x),:)];
    if (labels(r(x)) == 1) && (labels(c(x)) == 0)
        L1(x) = 1;  % a boundary exists here
        L2(x) = 0;  % wrong direction!
    elseif (labels(c(x)) == 1) && (labels(r(x)) == 0)
        L2(x) = 1;  % a boundary exists here
        L1(x) = 0;  % wrong direction!
    else
        L1(x) = 0;
        L2(x) = 0;
    end
end


% normalize the data 
disp('normalizing');
for x = 1:size(D,1)
    featureVector1(:,D(x,1):D(x,2)) = mat2gray(featureVector1(:,D(x,1):D(x,2)), limits(x,:));
    featureVector2(:,D(x,1):D(x,2)) = mat2gray(featureVector2(:,D(x,1):D(x,2)), limits(x,:));
end

% perform the SVM prediction
cmd = '-b 1';
disp('predicting 1/2');
[pre_L1, acc1, probs1] = svmpredict(L1, featureVector1, model, cmd); disp('predicting 2/2');
[pre_L2, acc2, probs2] = svmpredict(L2, featureVector2, model, cmd);


% determine centroid locations on adjacency graph
locs = zeros(length(superpixels), 2);
for s = superpixels
    locs(s,:) = STATS(s).Centroid;
end

% combine our 2 directional predictions
ind = find(model.Label == 1);       
probsCUT = max(probs1(:,ind), probs2(:,ind));
probsNOT = [ 1 - probsCUT];
if ind == 1
    probs = [probsCUT probsNOT];
else
    probs = [probsNOT probsCUT];
end


P = sparse([],[],[], size(A,1), size(A,2),0);
for x = 1:length(r)
    P(r(x),c(x)) = probs(x);
end
P = max(P,P');

% create and write the image
I = imread([imgFolder fileRoot '.png']);
% THRESHL = .5;    THRESHH = .85;
% hold off; figure(1); cla; imshow(I); hold on;
% gplot2(P > THRESHL ,locs, 'y-');
% gplot2(P > THRESHH, locs, 'r-');


hold off; figure(1); cla; imshow(I); hold on;
colors = jet(25); %17:24

gplot2(P > .5 ,locs, '.-', 'Color', colors(17,:));
gplot2(P > .5714 ,locs, '.-', 'Color', colors(18,:));
gplot2(P > .6429 ,locs, '.-', 'Color', colors(19,:));
gplot2(P > .7143 ,locs, '.-', 'Color', colors(20,:));
gplot2(P > .7857 ,locs, '.-', 'Color', colors(21,:));
gplot2(P > .8571 ,locs, '.-', 'Color', colors(22,:));
gplot2(P > .9286 ,locs, '.-', 'Color', colors(23,:));
gplot2(P > 1 ,locs, '.-', 'Color', colors(24,:));



%STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
%I = imread([imgFolder fileRoot '.png']);
%load([adjacencyFolder fileRoot '.mat']);







