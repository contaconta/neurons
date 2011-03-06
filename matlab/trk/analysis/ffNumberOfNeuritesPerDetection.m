function vret =  ffNumbreOfNeuritesPerDetection(E, varargin)

% By default look at happy neurons
LookOnlyAtHappyNeurons = 0;

optargin = size(varargin,2);
if optargin >= 1
   LookOnlyAtHappyNeurons = varargin{1};
end

vret = zeros(300000,1);

nCount = 1;
% for each neuron
for nTrack = 1:length(E.trkSeq)
    
    trck = E.trkSeq{nTrack};
    
    %If the track is empty, we should continue
    if(numel(trck) == 0)
        continue;
    end
    
    % If a neurong is interesting (happy), it is so during the whole track
    % checking only for the first value will then work
    if (LookOnlyAtHappyNeurons && E.D(trck(1)).Happy==0) % Shall we remove sad neurons from statistics?    
       continue;
    end
    
    for nNT = 1:length(trck)
        if ~isempty(E.FILAMENTS(trck(nNT)).NeuriteID)
        vret(nCount) = max(E.FILAMENTS(trck(nNT)).NeuriteID);
        else
                    vret(nCount) = 0;
        end
        nCount = nCount + 1;
    end
end

vret = vret(1:nCount);