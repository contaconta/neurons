function [FILAMENTS neuritePixList] = trkFindFilopodia(n,FILAMENTS)

MAX_LENGTH_FILOPODIA = 20;

neuritePixList = [];
%neuriteinds = [];
FILAMENTS(n).NeuriteFlag = zeros(size(FILAMENTS(n).PixelIdxList));

if ~isempty(FILAMENTS(n).NeuriteID)

    for  p = 1:length(FILAMENTS(n).PixelIdxList)
        nkids = FILAMENTS(n).NumKids(p);
        neuriteID = FILAMENTS(n).NeuriteID(p);
        if (nkids == 0) && (neuriteID > 0)
            plist = [];
            k = p;
            while (nkids <= 1) && (neuriteID > 0)
                plist = [plist; k]; %#ok<AGROW>
                k = FILAMENTS(n).Parents(k);
                nkids = FILAMENTS(n).NumKids(k);
                neuriteID = FILAMENTS(n).NeuriteID(k);
            end
            
        
            
            if length(plist) < MAX_LENGTH_FILOPODIA
                pixlist = FILAMENTS(n).PixelIdxList(plist);
                %neuriteinds = [neuriteinds; plist];
                FILAMENTS(n).FilopodiaFlag(plist) = 1;
                neuritePixList = [neuritePixList; pixlist]; %#ok<AGROW>
        	end
        end
    end    
end