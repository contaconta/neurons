function vret =  fnMeanNeuriteLength(R, varargin)


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
    
    mean_length_tstep = zeros(1,length(trck));
    for nNT = 1:length(trck)

        nDendrites = max(R.FILAMENTS(trck(nNT)).NeuriteID);
        %It could be a detection without dendrites then the mean is 0
        if isempty(nDendrites) || nDendrites == 0
            mean_length_tstep(nNT) = 0;
            continue;
        end
        % Computes the total dendrite length for all dendrites in the
        % neuron in the time step and divides by the number of dendrites
        total_neurite_length = length(find(R.FILAMENTS(trck(nNT)).NeuriteID > 0));
        if isempty(total_neurite_length)
            total_neurite_length = 0;
        end
        mean_length_tstep(nNT) = total_neurite_length/nDendrites;
    end
    nCount = nCount + 1;
    vret(nCount) = mean(mean_length_tstep);
end

vret = vret(1:nCount);