function [label,count] = getMostFrequentLabel(pixelList, IGroundTruth)

count = 0;
for p=1:size(pixelList,1)
    if(IGroundTruth(pixelList(p))==255)
        count = count + 1;
    else
        count = count - 1;
    end
end

if abs(count) < 0.3*size(pixelList,1)
    label = 2; % boundary
else
    if count < 0
        label = 1; % background
    else
        label = 3; % mitochondria
    end
end
