imgpath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/')

load featurevectors/rays30MedianInvariant/FIBSLICE0002.mat
I = imread([imgpath 'FIBSLICE0002.png']);

%STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');



% construct an Nxfeature_vector training vector
N = 350;
m = find(mito == 1);  % contains indexes of mitochondria-labeled superpixels
n = find(mito == 0);  % contains indexes of non-mitochondria-labeled superpixels
mlist = randsample(m, N)';
nlist = randsample(n, N)';
TRAIN = [RAYFEATUREVECTOR(mlist,:) ; RAYFEATUREVECTOR(nlist,:)];
TRAIN_L = [2*ones(size(mlist)); zeros(size(nlist))];


load featurevectors/rays30MedianInvariant/FIBSLICE0080.mat
m = find(mito == 1);  % contains indexes of mitochondria-labeled superpixels
n = find(mito == 0);  % contains indexes of non-mitochondria-labeled superpixels
mlist = randsample(m, N)';
nlist = randsample(n, N)';
TRAIN = [TRAIN; RAYFEATUREVECTOR(mlist,:) ; RAYFEATUREVECTOR(nlist,:)];
TRAIN_L = [TRAIN_L; 2*ones(size(mlist)); zeros(size(nlist))];

load featurevectors/rays30medianInvariant/FIBSLICE0160.mat
m = find(mito == 1);  % contains indexes of mitochondria-labeled superpixels
n = find(mito == 0);  % contains indexes of non-mitochondria-labeled superpixels
mlist = randsample(m, N)';
nlist = randsample(n, N)';
TRAIN = [TRAIN; RAYFEATUREVECTOR(mlist,:) ; RAYFEATUREVECTOR(nlist,:)];
TRAIN_L = [TRAIN_L; 2*ones(size(mlist)); zeros(size(nlist))];

load featurevectors/rays30medianInvariant/FIBSLICE0240.mat
I = imread([imgpath 'FIBSLICE0240.png']);
m = find(mito == 1);  % contains indexes of mitochondria-labeled superpixels
n = find(mito == 0);  % contains indexes of non-mitochondria-labeled superpixels
mlist = randsample(m, N)';
nlist = randsample(n, N)';
TRAIN = [TRAIN; RAYFEATUREVECTOR(mlist,:) ; RAYFEATUREVECTOR(nlist,:)];
TRAIN_L = [TRAIN_L; 2*ones(size(mlist)); zeros(size(nlist))];

% %model = svmtrain(TRAIN_L, TRAIN,'-t 0 -b 1');
%TRAIN = TRAIN(:,1:25);  RAYFEATUREVECTOR = RAYFEATUREVECTOR(:, 1:25);

% get rid of the grayscale data
TRAIN(:,1) = 0;


%% rescale the data
TRAIN_OLD = TRAIN;
limits1 = [min(min(TRAIN(:,1))) max(max(TRAIN(:,1)))];
limits2 = [min(min(TRAIN(:,2:13))) max(max(TRAIN(:,2:13)))];
limits14 = [min(min(TRAIN(:,14:25))) max(max(TRAIN(:,14:25)))];
limits26 = [min(min(TRAIN(:,26:37))) max(max(TRAIN(:,26:37)))];
limits38 = [min(min(TRAIN(:,38:103))) max(max(TRAIN(:,38:103)))];
TRAIN(:,1) = mat2gray(TRAIN(:,1), limits1);
TRAIN(:,2:13) = mat2gray(TRAIN(:,2:13), limits2);
TRAIN(:,14:25) = mat2gray(TRAIN(:,14:25), limits14);
TRAIN(:,26:37) = mat2gray(TRAIN(:,26:37), limits26);
TRAIN(:,38:103) = mat2gray(TRAIN(:,38:103), limits38);


%% TRAIN THE SVM

disp('Selecting parameters for the SVM');
bestcv = 0;
for log2c = -1:3,
  for log2g = -4:1,
    cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g) ' -m 500'];
    cv = svmtrain(TRAIN_L, TRAIN, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end


disp('Training the SVM');
cmd = ['-b 1 -c ' num2str(bestc) ' -g ' num2str(bestg) ' -m 500'];
model = svmtrain(TRAIN_L, TRAIN, cmd);
% model = svmtrain(TRAIN_L, TRAIN,'-t 0 -b 1 -m 500');


%% classify on the FIBSLICE0400
disp('Predicting');
load featurevectors/rays30MedianInvariant/FIBSLICE0400.mat
% get rid of the grayscale data
RAYFEATUREVECTOR(:,1) = 0;

RAYFEATUREVECTOR(:,1) = mat2gray(RAYFEATUREVECTOR(:,1), limits1);
RAYFEATUREVECTOR(:,2:13) = mat2gray(RAYFEATUREVECTOR(:,2:13), limits2);
RAYFEATUREVECTOR(:,14:25) = mat2gray(RAYFEATUREVECTOR(:,14:25), limits14);
RAYFEATUREVECTOR(:,26:37) = mat2gray(RAYFEATUREVECTOR(:,26:37), limits26);
RAYFEATUREVECTOR(:,38:103) = mat2gray(RAYFEATUREVECTOR(:,38:103), limits38);

I = imread([imgpath 'FIBSLICE0400.png']);
cmd = '-b 1';
[pre_L, acc, probs] = svmpredict(2*mito', RAYFEATUREVECTOR, model, cmd);


STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
P = zeros(size(I)); Pl = zeros(size(I));
mito_label_index = find(model.Label == 2);
for s = superpixels
    
    P(STATS(s).PixelIdxList) = probs(s,mito_label_index);
  	%Pl(STATS(s).PixelIdxList) = -2*log(probs(s,2)/probs(s,1));

 
end

% display the image
resultIM = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P)));
imshow(resultIM);

B = P > 0;
annotationpath = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
A = imread([annotationpath 'FIBSLICE0002' '.png']); A = A(:,:,2) > 200;
[ACC] = rocstats(B(:), A(:), 'ACC')

libsvmwrite('FIBSLICE0400.txt', 2*mito', sparse(probs(:,mito_label_index)));
