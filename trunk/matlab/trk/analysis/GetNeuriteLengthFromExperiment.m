function vlengths = GetNeuriteLengthFromExperiment(E, NeuriteThreshold)

if ~exist('NeuriteThreshold', 'var');
    NeuriteThreshold = 0;
end

vlengths = zeros(300000,1);
nCount = 1;

% for each neuron
for nTrack = 1:length(E.trkSeq)
    trck = E.trkSeq{nTrack};
    for nNT = 1:length(trck)
       for nDend = 1:1:max(E.FILAMENTS(trck(nNT)).NeuriteID)
            vlength = length(find(E.FILAMENTS(trck(nNT)).NeuriteID==nDend));
            if(vlength > NeuriteThreshold)
                vlengths(nCount) = vlength;
                nCount = nCount + 1;
            end
       end
    end
end

vlengths = vlengths(1:nCount);

