% Script to test the correctness of the likelood probability function.
% This script asks the user to click on a point in the image and returns the probabilities
% of this point to belong to each of the classes.

% set necessary paths
addpath('../utils/libsvm-mat-2.89-3');
addpath([pwd '/../bin/']);
addpath([pwd '/../utils/']);
listColors = ['g' 'r' 'c' 'm' 'y' 'k'];
%huttPath = '/osshare/Work/software/huttenlocher_segment/';
%imPath = [pwd '/../images/'];
%imPath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
imPath = '/home/alboot/usr/share/Data/LabelMe/Images/FIBSLICE/';
imName = 'FIBSLICE0720'
feature_vectors = '../temp/Model-0-4200-3-sup/feature_vectors';
%labelPath = '../temp/seg_plus_labels/';
labelPath = '../temp/labels/';
kernelType = '5'; % chi-square kernel
rescaleData = false;

[label_vector, instance_matrix] = libsvmread(feature_vectors);
training_label = label_vector(1:4000,:);
training_instance = instance_matrix(1:4000,:);
%testing_label = label_vector(3001:size(label_vector,1),:);
%testing_instance = instance_matrix(3001:size(instance_matrix,1),:);

if ~exist('model')
  disp('Computing model...');
  [model,minI,maxI] = loadModel(training_label, training_instance, rescaleData, kernelType, true);
end

disp('Model computed');

%Y = rand(1,100);
%img = imread('/home/alboot/usr/share/Data/LabelMe/Images/FIBSLICE/FIBSLICE0080.png');
%img = img(:,:,1);
%figure, imshow(img);
%h = imrect;
%position = wait(h)
% position has the form [xmin ymin width height]
%masked_img=img(position(2):position(2)+position(4), position(1):position(1)+position(3));
%imshow(masked_img);
%Y = masked_img(:);

% load an image we want to play with
Iraw = imread([imPath imName '.png']);

% load superpixels or atomic regions as a label matrix, L
%if ~exist('L')
%  HUT = imread([imPath 'FIBSLICE0100_superseg_010.jpg']);
%  L = rgb2label(HUT);
%end

labelFilenm = [labelPath imName '.dat'];
fid = fopen(labelFilenm,'r');
%FIXME :
%L = fread(fid,[size(Iraw,2) size(Iraw,1)],'int32');
L = fread(fid,[size(Iraw,2) size(Iraw,1)],'int32');
L = double(L);
L = L+1;
L = L';
fclose(fid);

figure(64);
imshow(Iraw);

while 1
  figure(64);
  [x,y] = ginput
  l = L(round(y),round(x));
  pixelList = find(L == l);

  [r,c] = ind2sub(size(L), pixelList);
  figure(65);
  hold on;
  plot(c,r,'b*');

  % Find neighbors
  BW = L==l;
  BW = bwmorph(BW,'dilate',1) - BW;
  BW(BW < 0) = 0;
  BW = logical(BW);
  neighbors = L(BW);
  neighbors = setdiff(unique(neighbors),l)';

  for n = neighbors
    pN = find(L == n);
    pixelList = [pixelList;pN];

    [r,c] = ind2sub(size(L), pN);
    col=listColors(round(rand(1)*length(listColors)));
    plot(c,r,[col '*']);
  end

  %clear Y;
  %I = Iraw(:);
  %Y = double(I(pixelList));
  % FIXME : Vector notation ?
  %for i=1:length(c)
  %  Y(i) = double(Iraw(r(i),c(i)));
  %end
  Y = double(Iraw(pixelList));
  [predicted_label, accuracy, pb, fv] = getLikelihood(Y, model,minI,maxI,rescaleData)
end

% Prediction
%n = testing_instance;
%T = (n - repmat(minI,size(n,1),1))*spdiags(1./(maxI-minI)',0,size(n,2),size(n,2));
%T(find(isnan(T)))=0;
%T(find(isinf(T)))=0;
%[predicted_label, accuracy, pb] = svmpredict(testing_label, T, model, '-b 1')
