function FILAMENTS = trkSkeletonize2(D, FIL, BLANK)

FILAMENTS = struct('PixelIdxList',[],'IMSIZE',[],'Parents',[],'NeuriteID',[],'NucleusID',[],'NumKids',[]);
FILAMENTS(length(D)).PixelIdxList = [];

% variable slicing for the parfor
fil = cell(size(D));
for d = 1:length(D)
    if D(d).ID ~= 0
        t = D(d).Time;
        fil{d} = FIL{t} == d;
    end
end

parfor d = 1:length(D)
    if D(d).ID ~= 0
        fil{d} = bwmorph(fil{d}, 'erode', 1); %#ok<PFOUS>
        FILSKEL = bwmorph(fil{d}, 'skel', Inf);
        FILAMENTS(d).PixelIdxList = find(FILSKEL);
        FILAMENTS(d).IMSIZE = size(BLANK);
        FILAMENTS(d).FilopodiaLengths = [];
        FILAMENTS(d).FilopodiaLengthMean = 0;
    end
end