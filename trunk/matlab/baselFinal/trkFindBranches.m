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
    leafBranchesLength                  = BranchesLength(logical(BranchesAreLeafs));
    filament.LeafLengthBranches         = leafBranchesLength;
    filament.ExtremeLength              = ExtremeLength;
    filament.MaxExtremeLength           = max(ExtremeLength);
    filament.MeanBranchLength           = mean(BranchesLength);
    
    filament.MeanLeafLength             = mean(leafBranchesLength);
    filament.TotalCableLength           = sum(BranchesLength);
    fieldsToQuantile = {'LengthBranches', 'ExtremeLength', 'LeafLengthBranches'};
    quantilesList    = [0 0.25, 0.5, 0.75 1];
    filament         = trkComputeQuantilesAndMean(filament, fieldsToQuantile, quantilesList);
    
    filament.Complexity =  length(filament.LengthBranches) / filament.TotalCableLength;
    if(isnan(filament.TotalCableLength) || isinf(filament.TotalCableLength) || filament.TotalCableLength <=0 )
        warning('on', 'problem with filaments total cable length');
        keyboard;
    end
    
else
    filament.Branches                   = [];
    filament.LeafBranches               = [];
    filament.LengthBranches             = [];
    filament.LeafLengthBranches         = [];
    filament.ExtremeLength              = [];
    filament.MaxExtremeLength           = nan;
    filament.MeanBranchLength           = nan;
    filament.MedianBranchLength         = nan;
    filament.TwentyFiveBranchLength     = nan;
    filament.SeventyFiveBranchLength    = nan;
    filament.MeanLeafLength             = nan;
    filament.MedianLeafLength           = nan;
    filament.TwentyFiveLeafLength       = nan;
    filament.SeventyFiveLeafLength      = nan;
    filament.TotalCableLength           = nan;
    filament.Complexity                 = nan;
    keyboard;
end

