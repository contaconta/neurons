function FILAMENTS = trkSkeletonize(D, FIL, BLANK)

%FILAMENTS = [];

FILAMENTS = struct('PixelIdxList',[]);
FILAMENTS(length(D)).PixelIdxList = [];
parfor d = 1:length(D)
    if D(d).ID ~= 0
        t = D(d).Time;
        FILi = FIL{t} == d; %#ok<PFBNS>
        FILi = bwmorph(FILi, 'erode', 1);
        FILSKEL = bwmorph(FILi, 'skel', Inf);
        FILAMENTS(d).PixelIdxList = find(FILSKEL);
        %FILAMENTS(d).Endpoints = find(bwmorph(FILSKEL, 'endpoints'));
        %%FILAMENTS(d).Branchpoints = find(bwmorph(FILSKEL, 'branchpoints'));
        %FILAMENTS(d).Branchpoints = find( bwmorph(bwmorph(FILSKEL, 'branchpoints'), 'thin', Inf));
        FILAMENTS(d).IMSIZE = size(BLANK);
    end
end