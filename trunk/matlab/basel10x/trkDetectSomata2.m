%% detect somata
function [Soma SL] = trkDetectSomata2(TMAX, Dlist, tracks, D, SOMA_THRESH, J)

BLANK = zeros(size(J{1}));
S = cell(1, length(D));

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


% prepare J1 for slicing
J1 = S;
for d = 1:length(D)
    J1{d} = J{D(d).Time};
end
clear d;

dmax = length(D);

parfor d = 1:dmax


    if D(d).ID ~= 0
        r = max(1,round(D(d).Centroid(2)));
        c = max(1,round(D(d).Centroid(1)));
        DET = BLANK;
        DET(D(d).PixelIdxList) = 1;
        DET = DET > 0;
        SOMA_INT_DIST =  SOMA_THRESH * mean(J1{d}(DET));

        % segment the Soma using region growing
        SomaM    = trkRegionGrow3(J1{d},DET,SOMA_INT_DIST,r,c);

        % fill holes in the somas, and find soma perimeter
        SomaM  	= imfill(SomaM, 'holes');
        SomaM   = bwmorph(SomaM, 'dilate', 2);

        % collect information about the soma region
        soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
        soma_prop(1).ID = D(d).ID;
        soma_prop(1).Time = D(d).Time;
        soma_prop(1).MeanGreenIntensity = sum(J1{d}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;

        % store properties into the Soma struct
        S{d} = soma_prop(1);
    else    
        S{d} = s;
    end 
end


for d = 1:length(D)
    Soma(d) = S{d}; %#ok<AGROW>
end


SL = cell(1, length(D));
for t = 1:TMAX
    SL{t} = BLANK;
    
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);
        
        if tracks(detect_ind) ~= 0
            SL{t}(S{detect_ind}.PixelIdxList) = detect_ind;
        end
    end
end




% parfor t = 1:TMAX
%     SMASK{t} = BLANK;
%     SL{t} = BLANK;
%     
%     J1 = J{t};
%     
%     for d = 1:length(Dlist{t})
%         detect_ind = Dlist{t}(d);
% 
%         r = max(1,round(D(detect_ind).Centroid(2)));
%         c = max(1,round(D(detect_ind).Centroid(1)));
%         DET = BLANK;
%         DET(D(detect_ind).PixelIdxList) = 1;
%         DET = DET > 0;
%         SOMA_INT_DIST =  SOMA_THRESH * mean(J1(DET));
% 
% 
%         % segment the Soma using region growing
%         SomaM    = trkRegionGrow3(J1,DET,SOMA_INT_DIST,r,c);
% 
%         % fill holes in the somas, and find soma perimeter
%         SomaM  	= imfill(SomaM, 'holes');
%         SomaM   = bwmorph(SomaM, 'dilate', 2);
% 
%         if tracks(detect_ind) ~= 0
%             % collect information about the soma region
%             soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
%             soma_prop(1).ID = tracks(detect_ind);
%             soma_prop(1).Time = t;
%             soma_prop(1).MeanGreenIntensity = sum(J{t}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;
% 
%             % store properties into the Soma struct
%             Soma(detect_ind) = soma_prop(1);
% 
%             % add the soma to a label mask
%             SL{t}(soma_prop(1).PixelIdxList) = detect_ind;
%         else
%             if detect_ind ~= 1
%                 Soma(detect_ind) = s;
%             else
%                 Soma = s;
%             end
%         end
% 
%         SMASK{t}(SomaM) = 1;
% 
%     end
% 
%     SMASK{t} = SMASK{t} > 0;
% end
