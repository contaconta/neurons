%% detect somata
function [Soma, S] = trkDetectSomataGrouped(M, J)




% dmax = length(D);
TMAX = length(M);

h = [1;1];
multFactor = 1.5;
S = cell(size(J));

parfor t = 1:TMAX
    mean_std = zeros(2, max(M{t}(:)));
    for i=1:max(M{t}(:))
       mean_std(1, i) = mean(J{t}(M{t} == i));
       mean_std(2, i) = std(J{t}(M{t} == i));
    end
    meanGlobal = mean(M{t}(:));
    stdGlobal  = std(M{t}(:));
    [U, V, L] = RegionGrowingSomata(h, J{t}, M{t}, mean_std, multFactor, meanGlobal, stdGlobal);
    SomaM  	= imfill((U < 0.000002) & (L < 7), 'holes');
    V(~SomaM) = 0;
    S{t} = V;
end
d = 1;
for t = 1:TMAX
    detections_t = regionprops(S{t}, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    if ~isempty(detections_t)
        for i = 1:length(detections_t)
            detections_t(i).MeanGreenIntensity = sum(J{t}(detections_t(i).PixelIdxList))/detections_t(i).Area;
            detections_t(i).Time = t;
            So{d} = detections_t(i);%#ok
            d = d+1;
        end
    end
end
clear Soma
for d =1:length(So)
   Soma(d) = So{d};%#ok
end