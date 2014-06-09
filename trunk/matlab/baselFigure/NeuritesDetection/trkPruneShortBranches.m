function [ PixelIndicesOfPrunedBranches ] = trkPruneShortBranches( filament, imSize, pruningThreshold )


PixelIndicesOfPrunedBranches = [];

for  p = 1:length(filament.NumKids)
    nkids = filament.NumKids(p);
    if (nkids ~= 1 && filament.Parents(p) > 0)% not a simple point and not a root
        plist = p;
        k = filament.Parents(p);
        if k > 0
            plist = [plist; k]; %#ok<AGROW>
            nkids = filament.NumKids(k);
        else% means that you arrived at the root of the tree
            % just assign a value so that you get out of the while loop
            nkids = 2;
        end
        while (nkids == 1) && k > 0
            k = filament.Parents(k);
            if k > 0
                plist = [plist; k]; %#ok<AGROW>
                nkids = filament.NumKids(k);
            else% means that you arrived at the root of the tree
                % just assign a value so that you get out of the while loop
                nkids = 2;
            end
        end
        
        leng = 0;
        [r c] = ind2sub(imSize, filament.NeuritePixelIdxList(plist) );
        
        for i=1:length(plist)-1;
            leng = leng + sqrt((r(i) - r(i+1))*(r(i) - r(i+1)) + (c(i) - c(i+1))*(c(i) - c(i+1)));
        end
        
        isThisBrancheAnEndingBranch = (filament.NumKids(plist(1)) == 0) || (filament.NumKids(plist(end)) ==0);
        
        if(isThisBrancheAnEndingBranch && leng < pruningThreshold)
            % don't cut the 
            PixelIndicesOfPrunedBranches = [PixelIndicesOfPrunedBranches (filament.NeuritePixelIdxList(plist(1:end-1)))']; %#ok
        end
        
        
    end
end
