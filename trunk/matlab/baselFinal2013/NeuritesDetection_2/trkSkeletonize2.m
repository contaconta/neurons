function [Cells, FILAMENTS] = trkSkeletonize2(Cells, FIL, BLANK)

FILAMENTS = struct('PixelIdxList',[],'IMSIZE',[],'Parents',[],'NeuriteID',[],'NucleusID',[],'NumKids',[]);
FILAMENTS(length(Cells)).PixelIdxList = [];

% variable slicing for the parfor
fil = cell(size(Cells));
for d = 1:length(Cells)
    if Cells(d).ID ~= 0
        t = Cells(d).Time;
        fil{d} = FIL{t} == d;
        Cells(d).Neurites                   = FIL{t};
    end
end

parfor d = 1:length(Cells)
    if Cells(d).ID ~= 0
        fil{d}  = bwmorph(fil{d}, 'erode', 1); %#ok<PFOUS>
        FILSKEL = bwmorph(fil{d}, 'skel', Inf);
        Cells(d).Neurites  = FILSKEL;
        FILAMENTS(d).PixelIdxList = find(FILSKEL);
        FILAMENTS(d).IMSIZE = size(BLANK);
    end
end