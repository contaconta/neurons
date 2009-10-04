% Script to test the correctness of the likelood probability function.
% This script outputs an image whose pixel values correspond to the
% class label given to each pixel.

% set necessary paths
addpath('../utils/libsvm-mat-2.89-3');
addpath([pwd '/../bin/']);
addpath([pwd '/../utils/']);
%huttPath = '/osshare/Work/software/huttenlocher_segment/';
%imPath = [pwd '/../images/'];
%imPath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
imPath = '~/usr/share/Data/LabelMe/Images/FIBSLICE/';
imName = 'FIBSLICE0720'
feature_vectors = '../temp/Model-0-4200-3-sup/feature_vectors';
%labelPath = '../temp/seg_plus_labels/';
labelPath = '../temp/labels/';
rescaleData = 1;

[label_vector, instance_matrix] = libsvmread(feature_vectors);
training_label = label_vector(1:4000,:);
training_instance = instance_matrix(1:4000,:);
%testing_label = label_vector(3001:size(label_vector,1),:);
%testing_instance = instance_matrix(3001:size(instance_matrix,1),:);

if ~exist('model')
  disp('Computing model...');
  [model,minI,maxI] = loadModel(training_label, training_instance,rescaleData,2);
end

disp('Model computed');

% load an image we want to play with
Iraw = imread([imPath imName '.png']);

% load superpixels or atomic regions as a label matrix, L
%if ~exist('L')
%  HUT = imread([imPath 'FIBSLICE0100_superseg_010.jpg']);
%  L = rgb2label(HUT);
%end

labelFilenm = [labelPath imName '.dat'];
size(Iraw,2)
size(Iraw,1)
L = readRKLabel(labelFilenm, [size(Iraw,1) size(Iraw,2)])';

% Only select part of the image (do not rescale before loading L)
L = L(1:480,1:640);
Iraw = Iraw(1:480,1:640);
outPb = zeros(size(L));

% extract and adjacency matrix and list from L
disp('Extracting adjacency graph G0 from superpixel segmentation image.');
[G0, G0list] = adjacency(L);

disp('Adjacency matrix extracted');
x0 = 1;
for x=x0:size(Iraw,2)
  x
  for y=1:size(Iraw,1)
    l = L(round(y),round(x));
    pixelList = find(L == l);

    neighbors = G0list{l};

    for n = neighbors
      pN = find(L == n);
      pixelList = [pixelList;pN];
    end

    %clear Y Y2;
    clear Y;
    %[r,c] = ind2sub(size(L), pixelList);
    %I = Iraw(:);
    %Y = double(I(pixelList));
    % FIXME : Vector notation ?
    %for i=1:length(c)
    %  Y(i) = double(Iraw(r(i),c(i)));
    %end
    Y = double(Iraw(pixelList));
    [predicted_label, accuracy, pb] = getLikelihood(Y, model,minI,maxI,rescaleData);

    outPb(y,x) = find(pb == max(pb),1);
    
  end
end

%figure;
%imagesc(outPb);
save outPb

% Prediction
%n = testing_instance;
%T = (n - repmat(minI,size(n,1),1))*spdiags(1./(maxI-minI)',0,size(n,2),size(n,2));
%T(find(isnan(T)))=0;
%T(find(isinf(T)))=0;
%[predicted_label, accuracy, pb] = svmpredict(testing_label, T, model, '-b 1')
