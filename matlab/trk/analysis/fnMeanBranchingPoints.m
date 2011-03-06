function vret =  fnMeanBranchingPoints(R, varargin)


% By default look at happy neurons
LookOnlyAtHappyNeurons = 0;

optargin = size(varargin,2);
if optargin >= 1
    LookOnlyAtHappyNeurons = varargin{1};
end

vret = zeros(5000,1);

nCount = 0;
% for each neuron
for nTrack = 1:length(R.trkSeq)
    trck = R.trkSeq{nTrack};
    
     %If the track is empty, we should continue
    if(numel(trck) == 0)
        continue;
    end
    
    % If a neurong is interesting (happy), it is so during the whole track
    % checking only for the first value will then work
    if (LookOnlyAtHappyNeurons && R.D(trck(1)).Happy == 0) % Shall we remove sad neurons from statistics?    
       continue;
    end   
    
    brpoints_tstep = zeros(1,length(trck));
    for nNT = 1:length(trck)

        % Computes the number of branching points
        % neuron in the time step and divides by the number of dendrites
        brpoints = length(find(R.FILAMENTS(trck(nNT)).NumKids >= 2));
        if isempty(brpoints)
            brpoints = 0;
        end
        brpoints_tstep(nNT) = brpoints;
    end
    nCount = nCount + 1;
    vret(nCount) = mean(brpoints_tstep);
end

vret = vret(1:nCount);