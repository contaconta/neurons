function [FILAMENTS neuritePixList] = trkFindFilopodia(n,FILAMENTS, Green)

MAX_LENGTH_FILOPODIA = 20;

neuritePixList = [];
%neuriteinds = [];

FILAMENTS(n).FilopodiaFlag = zeros(size(FILAMENTS(n).PixelIdxList));

%added by Fethallah
FILAMENTS(n).FilopodiaLengths = [];
FILAMENTS(n).FilopodiaF_Actin = [];
FILAMENTS(n).FilopodiaTotalF_Actin = 0;
FILAMENTS(n).FilopodiaMeanF_Actin  = 0;
FILAMENTS(n).FilopodiaMeanLengths  = 0;

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
                
                FILAMENTS(n).FilopodiaLengths = [FILAMENTS(n).FilopodiaLengths length(plist)];
                FILAMENTS(n).FilopodiaF_Actin = [FILAMENTS(n).FilopodiaF_Actin sum(Green(pixlist))];
            end
            
        end
    end
    if( ~isempty( FILAMENTS(n).FilopodiaLengths ) )
       FILAMENTS(n).FilopodiaMeanLengths  = mean( FILAMENTS(n).FilopodiaLengths );
       FILAMENTS(n).FilopodiaTotalF_Actin = sum(FILAMENTS(n).FilopodiaF_Actin);
       FILAMENTS(n).FilopodiaMeanF_Actin  = mean(FILAMENTS(n).FilopodiaF_Actin);
    end
end