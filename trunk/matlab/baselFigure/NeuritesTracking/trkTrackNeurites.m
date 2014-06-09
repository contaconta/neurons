function [TrackedNeurites, TrackedNeuritesList, trkNSeq, timeNSeq] = trkTrackNeurites(Cells, CellsList, timeSeq, NeuriteTrackingParams)


NEURITE_STABILITY_LENGTH_THRESHOLD = NeuriteTrackingParams.NEURITE_STABILITY_LENGTH_THRESHOLD;
W_THRESH                           = NeuriteTrackingParams.W_THRESH;
MIN_TRACK_LENGTH                   = NeuriteTrackingParams.MIN_TRACK_LENGTH;

% extracting neurites
[Neurites NeuritesList]  = getNeurites(Cells, CellsList, NEURITE_STABILITY_LENGTH_THRESHOLD);

% make and adjacency matrix of neurites
A = make_adjacency(NeuritesList, Neurites, timeSeq);

% fill out all the distances in the adjacency matrix
edges = find(A == 1);
W = A; 
for i = 1:length(edges)
    [r,c] = ind2sub(size(A), edges(i));
    W(r,c) = trkNeuriteDistance(Neurites(r), Neurites(c));
end

% apply the greedy tracking algorithm to link detections
T = trkGreedyConnect2(W,A,Neurites,W_THRESH);

% get the track labels from T assigned to each detection
[T ntracks] = trkGraphColoring(T, MIN_TRACK_LENGTH); clear T; %#ok<ASGLU>

% assign NeuriteTrack ID's to each neurite
for n = 1:length(Neurites)
    Neurites(n).NeuriteTrackId = ntracks(n);
end

TrackedNeurites      = Neurites;
TrackedNeuritesList  = NeuritesList;

% get a list of detections and associated time steps for each track
[trkNSeq, timeNSeq] = getTrackSequences(NeuritesList, ntracks, Neurites);

end

%% =========================================================================
function [Neurites NeuritesList] = getNeurites(Cells, CellsList, NEURITE_STABILITY_LENGTH_THRESHOLD)

NeuritesList = cell(size(CellsList));
Neurites     = [];

count = 1;

for i =1:length(Cells)
    if Cells(i).ID > 0
        for j =1:length(Cells(i).NeuritesList)
            if Cells(i).NeuritesList(j).TotalCableLength > NEURITE_STABILITY_LENGTH_THRESHOLD
                currentNeurite                      = Cells(i).NeuritesList(j);
                Neurites                            = [Neurites currentNeurite];%#ok
                NeuritesList{currentNeurite.Time}   = [NeuritesList{currentNeurite.Time} count];
                count = count + 1;
            end
        end
    end
end

end

%% =========================================================================
function A = make_adjacency(NeuritesList, Neurites, timeSeq)
Ndetection = length(Neurites);
A = zeros(Ndetection);
for t = 2:length(NeuritesList)
    for d = 1:length(NeuritesList{t})
        n_i         = NeuritesList{t}(d);              % neurite index
        nuc_i       = Neurites(n_i).CellTrackId;       % neurite's nucleus track
        time_i      = Neurites(n_i).Time;              % neurite frame time
        indexOfNextTime = find(timeSeq{nuc_i} == time_i);
        if(indexOfNextTime > 1)
            indexOfPreviousTime = indexOfNextTime - 1;
            time_j              = timeSeq{nuc_i}(indexOfPreviousTime);
            for dp = 1:length(NeuritesList{time_j})
                n_p     = NeuritesList{time_j}(dp);    % past neurite index
                nuc_p   = Neurites(n_p).CellTrackId;   % past neurite's nucleus
                if (nuc_i == nuc_p)
                    A(n_i, n_p) = 1;
                end
            end    
        end
    end
end
end

%%=========================================================================
%% extract series of track labels and time stamps for each valid track
function [trkSeq, timeSeq] = getTrackSequences(Nlist, tracks, N)
trkSeq = cell(1, max(tracks(:)));
timeSeq = cell(1, max(tracks(:)));
for i = 1:max(tracks(:))

    for t = 1:length(Nlist)
        detections = Nlist{t};
        ids = [N(detections).NeuriteTrackId];

        n = detections(find(ids == i,1));

        if ~isempty(n)
            trkSeq{i} = [trkSeq{i} n];
            timeSeq{i} = [timeSeq{i} t];
        end
    end
end
end
