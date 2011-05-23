close all; clc;
%% Input Image
%filename = '/net/cvlabfiler1/cvlab/Sinergia/Alan/2010-10-18/RMS3/101018RMS3_16-20-16_PMT_TP003.ome.tif';
filename = '/home/ksmith/Dropbox/fethallah/Soma_n_GrowthCone_Segmentation/101018RMS3_16-20-16_PMT_TP003.ome.tif';
InputImage = readMultiPageTiff(filename);

%% user has to provide the image spacing 
ImageSpacing = [1, 1, 3];

%% Crop Da image
croppedImage = double(InputImage(439:524, 393:511, :));

seedPoint = [44, 44, 29];
radius = 30;
%% sigmoid params
alpha = -1.0;
beta  = 20;
%% perform segmentation
tic
segmentation = SGAC(croppedImage, ImageSpacing, seedPoint, radius, alpha, beta);
toc