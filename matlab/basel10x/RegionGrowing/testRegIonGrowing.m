clc;
% 
% tic;[Dwin] = bwdist(M{imageIdx} > 0,'euclidean');toc
% %'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength',
% %'Orientation', 'Perimeter',
% detections_t = regionprops(M{imageIdx}, 'Area', 'PixelIdxList');  %#ok<*MRPBW>
% II = cell(1, length(detections_t));
% for i = 1:length(detections_t)
%     detections_t(i).MeanGreenIntensity = sum(J{1}(detections_t(i).PixelIdxList))/detections_t(i).Area;
%     II{i} = zeros(size(M{1}));
%     II{i}(detections_t(i).PixelIdxList) = detections_t(i).MeanGreenIntensity;
% end
%%
load detections;
% addpath('Release/')
%%
imageIdx = 1;
h = [1;1];
mean_std = zeros(2, max(M{imageIdx}(:)));
for i=1:max(M{imageIdx}(:))
   mean_std(1, i) = mean(J{imageIdx}(M{imageIdx} == i));
   mean_std(2, i) = std(J{imageIdx}(M{imageIdx} == i));
end
multFactor = 1.5;
meanGlobal = mean(M{imageIdx}(:));
stdGlobal  = std(M{imageIdx}(:));
tic
[U, V, L] = RegionGrowingSomata(h, J{imageIdx}, M{imageIdx}, mean_std, multFactor, meanGlobal, stdGlobal);
toc
%%
load Soma_tubularity;
tic
[U, V, L] = RegionGrowingNeurites(h, f{imageIdx}, double(S{imageIdx}));
toc