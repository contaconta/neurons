function [ filament ] = trkFindBranches(d,n, FILAMENTS)


filament = FILAMENTS(d);

nIdx =(filament.NeuriteID == n);

%pixelsIdxList = filament.PixelIdxList(nIdx);

NeuriteBranches = {};
BranchesLength = [];

%
nIdx = find(nIdx);


if ~isempty(filament.NeuriteID)
    FilopodiaFlags = [];
    for  p = 1:length(nIdx)
        nkids = filament.NumKids(nIdx(p));
        neuriteID = filament.NeuriteID(nIdx(p));
        if (nkids ~= 1) && (neuriteID > 0)
            k = nIdx(p);
            FilopodiaFlags = [FilopodiaFlags, filament.FilopodiaFlag(k)];%#ok<AGROW>
            
            plist = k;
            k = filament.Parents(k);
            nkids = filament.NumKids(k);
            neuriteID = filament.NeuriteID(k);
            plist = [plist; k]; %#ok<AGROW>
            
            
            
            
            while (nkids == 1) && (neuriteID > 0)
                k = filament.Parents(k);
                plist = [plist; k]; %#ok<AGROW>
                nkids = filament.NumKids(k);
                neuriteID = filament.NeuriteID(k);
            end
            NeuriteBranches = [NeuriteBranches; plist];%#ok<AGROW>
            
            
            leng = 0;
            [r c] = ind2sub(filament.IMSIZE, filament.PixelIdxList(plist) );
            
            for i=1:length(plist)-1;
                leng = leng + sqrt((r(i) - r(i+1))*(r(i) - r(i+1)) + (c(i) - c(i+1))*(c(i) - c(i+1)));
            end
            
            BranchesLength = [BranchesLength, leng];%#ok<AGROW>
        end
    end
    if( ~isempty( BranchesLength ) )
       filament.NeuriteBranches = NeuriteBranches;
       filament.FilopodiaFlags  = FilopodiaFlags;
       filament.BranchLengthsDistribution = BranchesLength;
       filament.MeanBranchLength = mean(BranchesLength);
       filament.FethTotalCableLength = sum(BranchesLength);
       filament.FethTotalCableLengthWithoutFilopodia  = sum(BranchesLength(~logical(FilopodiaFlags)));
       if(~isempty(~logical(FilopodiaFlags)))
           filament.MeanBranchLengthWithoutFilo = mean(BranchesLength(~logical(FilopodiaFlags)));
       end
    else
       filament.NeuriteBranches = [];
       filament.FilopodiaFlags  = [];
       filament.BranchLengthsDistribution = [];
       filament.MeanBranchLength = 0;
       filament.MeanBranchLengthWithoutFilo = 0;
       filament.FethTotalCableLength = 0;
       filament.FethTotalCableLengthWithoutFilopodia  = 0;
    end
end


% for i = 1:length(branchingPoints)
%     %numberOfKids = filament.NumKids(nIdx(branchingPoints(i)));
%     kidsPixelIdx = filament.PixelIdxList(nIdx(filament.Parents(nIdx) == nIdx(branchingPoints(i))));
%     kids = nIdx(filament.Parents(nIdx) == nIdx(branchingPoints(i)));
%     for k=1:length(kids)
%         branche = [nIdx(branchingPoints(i))];
%         current = kids(k);
%         nkids = filament.NumKids(current);
%         neuriteID = filament.NeuriteID(current);
%         if (nkids == 0) && (neuriteID > 0)
%             k = current;
%             while (nkids <= 1) && (neuriteID > 0)
%                 branche = [branche; k]; %#ok<AGROW>
%                 k = filament.Parents(k);
%                 nkids = filament.NumKids(k);
%                 neuriteID = filament.NeuriteID(k);
%             end
%         end
%         
%     end
% end

