% Script to test the correctness of the KL divergence between 2 edges.

if ~exist('labelPath','var')
% set necessary paths
addpath([pwd '/../bin/']);
addpath([pwd '/../utils/']);
imPath = '~/usr/share/Data/LabelMe/Images/FIBSLICE/';
imName = 'FIBSLICE0720'
labelPath = '../temp/labels/';

% load an image we want to play with
Iraw = imread([imPath imName '.png']);

% load superpixels or atomic regions as a label matrix, L
%if ~exist('L')
%  HUT = imread([imPath 'FIBSLICE0100_superseg_010.jpg']);
%  L = rgb2label(HUT);
%end

labelFilenm = [labelPath imName '.dat'];
L = readRKLabel(labelFilenm, [size(Iraw,1) size(Iraw,2)])';

% Only select part of the image (do not rescale before loading L)
L = L(1:480,1:640);
Iraw = Iraw(1:480,1:640);

% extract and adjacency matrix and list from L
if ~exist('G0','var')
  disp('Extracting adjacency graph G0 from superpixel segmentation image.');
  [G0, G0list] = adjacency(L);
else
    disp('Adjacency graph G0 was already computed.');
end

% create a list of superpixel center locations and pixel lists
disp('Computing superpixel center locations.');
centers = zeros(max(L(:)),1); pixelList = cell([1 size(G0,1)]);
for l = 1:max(L(:))
    pixelList{l} = find(L == l); 
    [r,c] = ind2sub(size(L), pixelList{l});
    centers(l,1) = mean(r);
    centers(l,2) = mean(c);
end

% precompute the KL divergences
disp('Precomputing the KL divergences.');
KL = edgeKL(Iraw, pixelList, G0, 1);
end

%gplotl(G0,centers,LABELS,Iraw);
imshow(Iraw);
hold on;
plot(centers(:,2),centers(:,1),'*')
%axis([0 480 0 640]);

while 1
	[x,y] = ginput(2);
	l1 = L(y(1),x(1));
	l2 = L(y(2),x(2));

	G0(l1,l2)
	KL(l1,l2)
end

