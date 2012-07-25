function [Rblur,  J, D, Dlist, count] = preprocessImagesMSER(R, G, sigma_log_red, minArea, maxArea, D, Dlist, count)


TMAX = length(R);
J = cell(size(R));
M = cell(size(R));
Rblur = cell(size(R));
EightBitsImages = cell(size(R));
%%
tic
parfor  t = 1:TMAX
    Rt = mat2gray(double(R{t}));
    J{t} = mat2gray(double(G{t}));
    Rblur{t} = imgaussian(Rt, sigma_log_red);
    I = Rblur{t};
    I = uint8(255*(I-min(I(:)))/(max(I(:)) - min(I(:))));
    EightBitsImages{t} = I;
    M{t} = vl_mser(I, 'MinDiversity', 0.5,...
        'MaxVariation', 0.25,...
        'MinArea', minArea/numel(I), ...
        'MaxArea', maxArea/numel(I), ...
        'BrightOnDark', 1, ...
        'Delta',2) ;
end
dt = toc;
disp(['computation time for preprocessing and nuclei detection with MSER ' num2str(dt)]);
%%
% idx = 13;
% Ellipses{idx} = vl_ertr(Ellipses{idx}) ;
% %%
% figure; imshow(max(Rblur{idx}(:)) - Rblur{idx}, []);
% hold on;
% vl_plotframe(Ellipses{idx}) ;
% %%
% figure; imshow(R{idx}, []);
% hold on;
% vl_plotframe(Ellipses{idx}) ;
%%
for t = 1:TMAX
    
    mm = zeros(size(R{1}));
    for x = M{t}'
        s = vl_erfill(EightBitsImages{t}, x);
        mm(s) = mm(s)+1;
    end
    M{t} = mm;
    L = bwlabel(M{t});
    detections_t = regionprops(L, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    
    
    % add some measurements, create a list of detections
    if ~isempty(detections_t)
        for i = 1:length(detections_t)
            if detections_t(i).Eccentricity < 0.85
                detections_t(i).MeanGreenIntensity = sum(G{t}(detections_t(i).PixelIdxList))/detections_t(i).Area;
                detections_t(i).MeanRedIntensity = sum(R{t}(detections_t(i).PixelIdxList))/detections_t(i).Area;
            
                detections_t(i).Time = t;
                if count == 1
                    D = detections_t(i);
                else
                    D(count) = detections_t(i);
                end
                Dlist{t} = [Dlist{t} count];
                count = count + 1;
            end
        end
    end
end