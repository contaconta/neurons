function vlengths = ffNeuriteLength(E, varargin)

optargin = size(varargin,2);

NeuriteThreshold = 0;
if optargin > 1
    NeuriteThreshold = varargin{1}
end

if ~exist('NeuriteThreshold', 'var');

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

