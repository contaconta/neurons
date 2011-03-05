function vret =  ffBranchingPointsPerNeurite(E, varargin)


optargin = size(varargin,2);
NeuriteThreshold = 0;
if optargin > 1
    NeuriteThreshold = varargin{1}
end

vret = zeros(300000,1);

nCount = 1;
% for each neuron
for nTrack = 1:length(E.trkSeq)
    trck = E.trkSeq{nTrack};
    for nNT = 1:length(trck)
       for nDend = 1:1:max(E.FILAMENTS(trck(nNT)).NeuriteID)
            vlength = length(find(E.FILAMENTS(trck(nNT)).NeuriteID==nDend));
            if(vlength > NeuriteThreshold)
                nBranches = length( find( E.FILAMENTS(trck(nNT)).NumKids >= 2) );
                vret(nCount) = nBranches;
                nCount = nCount + 1;
            end
       end
    end
end

vret = vret(1:nCount);