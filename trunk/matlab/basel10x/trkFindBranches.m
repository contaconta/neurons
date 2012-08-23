function [ filament ] = trkFindBranches( filament )

% minimalEndingBranchLength = 5;

NeuriteBranches = {};
BranchesLength = [];
%
nIdx =(filament.NeuriteId >=0);
nIdx = find(nIdx);

if ~isempty(filament.NeuriteId)
    for  p = 1:length(nIdx)
        nkids = filament.NumKids(nIdx(p));
        NeuriteId = filament.NeuriteId(nIdx(p));
        if (nkids ~= 1) && (NeuriteId >= 0)
            k = nIdx(p);
            plist = k;
            k = filament.Parents(k);
            nkids = filament.NumKids(k);
            NeuriteId = filament.NeuriteId(k);
            plist = [plist; k]; %#ok<AGROW>
            
            
            
            
            while (nkids == 1) && (NeuriteId >= 0)
                k = filament.Parents(k);
                plist = [plist; k]; %#ok<AGROW>
                nkids = filament.NumKids(k);
                NeuriteId = filament.NeuriteId(k);
            end
            
            leng = 0;
            [r c] = ind2sub(filament.IMSIZE, filament.NeuritePixelIdxList(plist) );
            
            for i=1:length(plist)-1;
                leng = leng + sqrt((r(i) - r(i+1))*(r(i) - r(i+1)) + (c(i) - c(i+1))*(c(i) - c(i+1)));
            end
            
%             isThisBrancheAnEndingBranch = (filament.NumKids(plist(1)) == 1) || (filament.NumKids(plist(end)) ==1);
%             if leng > minimalEndingBranchLength && isThisBrancheAnEndingBranch
                BranchesLength = [BranchesLength, leng];%#ok<AGROW>
                NeuriteBranches = [NeuriteBranches; plist];%#ok<AGROW>
%             else
%                 keyboard;
%             end
        end
    end
    if( ~isempty( BranchesLength ) )
       filament.NeuriteBranches           = NeuriteBranches;
       filament.BranchLengthsDistribution = BranchesLength;
       filament.MeanBranchLength          = mean(BranchesLength);
       filament.TotalCableLength          = sum(BranchesLength);
    else
       filament.NeuriteBranches           = [];
       filament.BranchLengthsDistribution = [];
       filament.MeanBranchLength          = 0;
       filament.TotalCableLength          = 0;
    end
end
