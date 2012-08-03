%% detect somata
function Soma = trkDetectSomata(D, SOMA_THRESH, J)

BLANK = zeros(size(J{1}));
S = cell(1, length(D));


% prepare J1 for slicing
J1 = S;
for d = 1:length(D)
    J1{d} = J{D(d).Time};
end
clear d;

dmax = length(D);

parfor d = 1:dmax
        r = max(1,round(D(d).Centroid(2)));
        c = max(1,round(D(d).Centroid(1)));
        DET = BLANK;
        DET(D(d).PixelIdxList) = 1;
        DET = DET > 0;
        SOMA_INT_DIST =  SOMA_THRESH * mean(J1{d}(DET));

        % segment the Soma using region growing
        SomaM = trkRegionGrow3(J1{d},DET,SOMA_INT_DIST,r,c);

        % fill holes in the somas, and find soma perimeter
        SomaM  	= imfill(SomaM, 'holes');
        SomaM   = bwmorph(SomaM, 'dilate', 2);

        % collect information about the soma region
        soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
        soma_prop(1).Time = D(d).Time;
        soma_prop(1).MeanGreenIntensity = sum(J1{d}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;
        % store properties into the Soma struct
        SomaM = [];%#ok
        S{d} = soma_prop(1);
end

for d = 1:length(D)
    Soma(d) = S{d}; %#ok<AGROW>
end