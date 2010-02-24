%%
clear all; close all; clc;
%--------------------------------------------------------------------------
data_directory   = '../data/';
addpath(data_directory);
matlab_directory   = '../matlab/';
addpath(matlab_directory);
mex_directory = '../mex/';
addpath(mex_directory);
%%
% Im = double(imread('LV.png'));
Im = double(imread('/media/data/steerableFilters2D/neurons/n7/2/N7_2.jpg'));
Im = sum(Im, 3);
Im = rescale(Im, 0, 1);
%%
R = [1:10];
h = [1;1];
%%
tic
[TT] = ScaledHessianGaussian2D([1;1], Im, R);
toc
%%
Pr = squeeze(TT(:,:, 1, :) + TT(:,:, 2, :));
Pmax = max(Pr,[], 3);
%%
figure;
imshow(Pmax, []);
colormap(jet);
colorbar;
set(gca, 'color', 'none');
print2im('Enhanced_OOF');