%% detect somata
function [Soma SMASK SL] = trkDetectSomata(TMAX, Dlist, tracks, D, SOMA_THRESH, J)

BLANK = zeros(size(J{1}));
Soma = [];
SL = [];

s.Area = []; 
s.Centroid = []; 
s.MajorAxisLength = []; 
s.MinorAxisLength = []; 
s.Eccentricity = []; 
s.Orientation = []; 
s.PixelIdxList = []; 
s.Perimeter = []; 
s.ID = []; 
s.Time = []; 
s.MeanGreenIntensity = []; 


for t = 1:TMAX
    SMASK{t} = BLANK;
    SL{t} = BLANK;
    
    J1 = J{t};
    
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        r = max(1,round(D(detect_ind).Centroid(2)));
        c = max(1,round(D(detect_ind).Centroid(1)));
        DET = BLANK;
        DET(D(detect_ind).PixelIdxList) = 1;
        DET = DET > 0;
        SOMA_INT_DIST =  SOMA_THRESH * mean(J1(DET));


        % segment the Soma using region growing
        SomaM    = trkRegionGrow3(J1,DET,SOMA_INT_DIST,r,c);

        % fill holes in the somas, and find soma perimeter
        SomaM  	= imfill(SomaM, 'holes');
        SomaM   = bwmorph(SomaM, 'dilate', 2);

        if tracks(detect_ind) ~= 0
            % collect information about the soma region
            soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
            soma_prop(1).ID = tracks(detect_ind);
            soma_prop(1).Time = t;
            soma_prop(1).MeanGreenIntensity = sum(J{t}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;
            %soma_prop(1).MeanRedIntensity = sum(R{t}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;

            % fill the soma structure
            if isempty(Soma)
                Soma = soma_prop(1);
            end

            % store properties into the Soma struct
            Soma(detect_ind) = soma_prop(1);

            % add the soma to a label mask
            SL{t}(soma_prop(1).PixelIdxList) = detect_ind;
        else
            if detect_ind ~= 1
                Soma(detect_ind) = s;
            else
                Soma = s;
            end
        end

        SMASK{t}(SomaM) = 1;

    end

    SMASK{t} = SMASK{t} > 0;
    %     SMASK{t} = bwmorph(SMASK{t}, 'dilate', 2);
end
