function [Soma SL] = trkDetectSomataWithMorphologicalSnakes(TMAX, Dlist, tracks, D, M, G)

BLANK = zeros(size(G{1}));
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

WIN = 50;
% prepare J1 for slicing
J1 = S;
for d = 1:length(D)
    J1{d} = G{D(d).Time};
end
clear d;

dmax = length(D);

for d = 1:dmax

    if D(d).ID ~= 0
        

        r = max(1,round(D(d).Centroid(2)));
        c = max(1,round(D(d).Centroid(1)));
        
        
        rmin = max(1, r - WIN);
        rmax = min(size(M{1},1), r + WIN);
        cmin = max(1, c - WIN); 
        cmax = min(size(M{1},2), c + WIN);

        Dwin = uint8(255*mat2gray(double(G{D(d).Time}(rmin:rmax,cmin:cmax))));
        DwinRGB = zeros(size(Dwin, 1), size(Dwin, 2), 3, 'uint8');
        for k =1:3
            DwinRGB(:,:, k) = Dwin;
        end
        subImagefilename = ['tmp/' sprintf('%08d', d) '.bmp'];
        subImageOutfilename = ['tmp/' sprintf('%08d', d) 'Out.bmp'];
        imwrite(DwinRGB, subImagefilename);
        
        Nucleus = M{D(d).Time}(rmin:rmax,cmin:cmax);
        Nucleus = bwmorph(Nucleus, 'dilate');
        Nucleus = bwmorph(Nucleus, 'thin',1) - M{D(d).Time}(rmin:rmax,cmin:cmax);
        listOfPoints = find(Nucleus > 0);
        listOfPoints = [listOfPoints(1) listOfPoints(floor(length(listOfPoints)/2)) listOfPoints(end)];
        [PointsI, PointsJ] = ind2sub(size(Nucleus), listOfPoints);
        TAB = [PointsJ; PointsI]';
        contourFileName = ['tmp/' sprintf('%08d', d) '.cn'];
        contourOutFileName = ['tmp/' sprintf('%08d', d) 'Out.cn'];
        dlmwrite(contourFileName, TAB, 'delimiter', ' ');
        
        cmd = ['/home/fbenmans/Downloads/MorphologicalSnakes_basic_20111228/source/morphological_snake' ...
                ' -I ' subImagefilename ...
                ' -C ' contourFileName ...
                ' -O ' subImageOutfilename ...
                ' -F ' contourOutFileName ...
                ' -T 1 -B 1 clc'];
        system(cmd);
        
        
        DET = BLANK;
        DET(D(d).PixelIdxList) = 1;
        DET = DET > 0;
        SOMA_THRESH = 100;
        SOMA_INT_DIST =  SOMA_THRESH * mean(J1{d}(DET));

        % segment the Soma using region growing
        SomaM = trkRegionGrow3(J1{d},DET,SOMA_INT_DIST,r,c);

        % fill holes in the somas, and find soma perimeter
        SomaM  	= imfill(SomaM, 'holes');
        SomaM   = bwmorph(SomaM, 'dilate', 2);

        % collect information about the soma region
        soma_prop = regionprops(SomaM, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
        soma_prop(1).ID = D(d).ID;
        soma_prop(1).Time = D(d).Time;
        soma_prop(1).MeanGreenIntensity = sum(J1{d}(soma_prop(1).PixelIdxList))/soma_prop(1).Area;
        % store properties into the Soma struct
        SomaM = [];%#ok
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