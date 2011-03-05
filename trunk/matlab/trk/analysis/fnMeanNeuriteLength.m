function vret =  fnMeanNeuriteLength(E, varargin)


optargin = size(varargin,2);
NeuriteThreshold = 0;
if optargin > 1
    NeuriteThreshold = varargin{1}
end

vret = zeros(300000,1);

nCount = 0;
% for each neuron
for nTrack = 1:length(E.trkSeq)
    trck = E.trkSeq{nTrack};
    mean_length_tstep = zeros(1,length(trck));
    for nNT = 1:length(trck)
        nDendrites = max(E.FILAMENTS(trck(nNT)).NeuriteID);
        %It could be a detection without dendrites
        if isempty(nDendrites)
            continue;
        end
        total_dendrite_length = 0;
        for nDend = 1:1:nDendrites
            total_dendrite_length = total_dendrite_length + ...
                length(find(E.FILAMENTS(trck(nNT)).NeuriteID == nDend));            
        end
        mean_length_tstep(nNT) = total_dendrite_length/nDendrites;
    end
    nCount = nCount + 1;
    vret(nCount) = mean(mean_length_tstep);
end

vret = vret(1:nCount);