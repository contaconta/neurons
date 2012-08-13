function [Cells, tracks, trkSeq, timeSeq] = trkGenerateNucleiGraphAndTrack(CellsList, Cells, WIN_SIZE, WT, WSH, W_THRESH, MIN_TRACK_LENGTH, SPATIAL_DIST_THRESH)

TMAX  = length(CellsList);
%% create the adjacency matrix for all nearby detections

Ndetection = length(Cells);
A = make_adjacency(CellsList, Cells, WIN_SIZE, SPATIAL_DIST_THRESH, Ndetection);

% fill out all the distances in the adjacency matrix
edges = find(A == 1);
W = edges;
parfor i =1:length(edges)
    [r,c] = ind2sub(size(A), edges(i));
    W(i) = trkDetectionDistanceNucleusAndSomata(Cells(r), Cells(c), WT, WSH);%#ok
end
WW = A;
WW(edges) = W;
W = WW;


%% apply the greedy tracking algorithm to link detections
T = trkGreedyConnect2(W,A,Cells,W_THRESH);


%% get the track labels from T assigned to each detection
[T tracks] = trkGraphColoring(T, MIN_TRACK_LENGTH); %#ok<*ASGLU>


%% assign ID's to each detections
for t = 1:TMAX
    for d = 1:length(CellsList{t}) % loop through detections in this time step
        detect_ind = CellsList{t}(d);
        Cells(detect_ind).ID = tracks(detect_ind); %#ok<*AGROW>
    end
end


%% get a list of detections and associated time steps for each track
[trkSeq, timeSeq] = getTrackSequences(CellsList, tracks, Cells);
