function [ filament ] = trkFindBranches( filament, imSize )

% minimalEndingBranchLength = 5;

NeuriteBranches     = [];
BranchesAreLeafs    = [];
BranchesLength      = [];
% get the branches

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
        BranchesLength      = [BranchesLength, leng];%#ok<AGROW>
        NeuriteBranches     = [NeuriteBranches; plist];%#ok<AGROW>
        BranchesAreLeafs    = [BranchesAreLeafs; isThisBrancheAnEndingBranch];%#ok<AGROW>
        
    end
end
% get the extreme lengths
ExtremeLength = [];
for  p = 1:length(filament.NumKids)
    nkids = filament.NumKids(p);
    if (nkids == 0)% not a simple point and not a root
        plist = p;
        par = filament.Parents(p);
        
        while par > 0 % until reaching the root
            plist = [plist; par]; %#ok<AGROW>
            par = filament.Parents(par);
        end
        
        leng = 0;
        [r c] = ind2sub(imSize, filament.NeuritePixelIdxList(plist) );
        
        for i=1:length(plist)-1;
            leng = leng + sqrt((r(i) - r(i+1))*(r(i) - r(i+1)) + (c(i) - c(i+1))*(c(i) - c(i+1)));
        end
        ExtremeLength = [ExtremeLength, leng];%#ok<AGROW>
    end
end

if( ~isempty( BranchesLength ) )
    filament.Branches                   = NeuriteBranches;
    filament.LeafBranches               = logical(BranchesAreLeafs');
    filament.LengthBranches             = BranchesLength;
    filament.ExtremeLength              = ExtremeLength;
    filament.MaxExtremeLenght           = max(ExtremeLength);
    filament.MeanBranchLength           = mean(BranchesLength);
    filament.MeanLeafLength             = mean(BranchesLength(logical(BranchesAreLeafs)));
    filament.TotalCableLength           = sum(BranchesLength);
else
    filament.Branches                   = [];
    filament.LeafBranches               = [];
    filament.LengthBranches             = [];
    filament.ExtremeLength              = [];
    filament.MaxExtremeLenght           = 0;
    filament.MeanBranchLength           = 0;
    filament.MeanLeafLength             = 0;
    filament.TotalCableLength           = 0;
end

