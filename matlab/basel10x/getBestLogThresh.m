% estimate the best performing threshold for the LapL nuclei detector
function BEST_LOG_THRESH = getBestLogThresh(log1, NUC_MIN_AREA, TARGET_NUM_OBJECTS)



thresh_list = -.0007:.00005:-.00005;
%Blap = zeros(size(log1{1},1),size(log1{1},2));

for j = 1:length(thresh_list)
    tcount = 1;
    for t = 1:10:length(log1)
        Thresh = thresh_list(j);
        Blap = log1{t} <  Thresh;
        Blap = bwareaopen(Blap, NUC_MIN_AREA);
        Bprop = regionprops(Blap, 'Eccentricity', 'PixelIdxList');
        for i = 1:length(Bprop)
            if Bprop(i).Eccentricity > .85
                Blap(Bprop(i).PixelIdxList) = 0;
            end
        end
        L = bwlabel(Blap);

        det_table(j,tcount) = max(max(L));
        tcount = tcount + 1;
    end

end

dists = abs(mean(det_table,2) - TARGET_NUM_OBJECTS);
[min_val, best_ind] = min(dists);

BEST_LOG_THRESH = thresh_list(best_ind);
disp(['...selected BEST_LOG_THRESH = ' num2str(BEST_LOG_THRESH)]);
