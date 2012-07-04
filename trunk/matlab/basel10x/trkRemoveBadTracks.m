function [D tracks trkSeq timeSeq] = trkRemoveBadTracks(D, tracks, trkSeq, timeSeq, MAX_NUCLEUS_AREA)


for t = 1:length(trkSeq)
   
    dseq = trkSeq{t};
    
    if ~isempty(dseq)
        arealist = [D(dseq).Area];
        meanarea = mean(arealist);
        
        if meanarea > MAX_NUCLEUS_AREA
            trkSeq{t} = []; 
            timeSeq{t} = [];
            
            for d = dseq
                D(d).ID = 0;
                tracks(d) = 0;
                
            end
            
            disp('REMOVED A TRACK WITH EXCESSIVELY LARGE NUCLEUI!')
        end
    end
end
