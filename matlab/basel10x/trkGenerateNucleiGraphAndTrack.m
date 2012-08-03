function [D, T, tracks, trkSeq, timeSeq] = trkGenerateNucleiGraphAndTrack(count, Dlist, D, WIN_SIZE, WT, WSH, W_THRESH, MIN_TRACK_LENGTH, SPATIAL_DIST_THRESH)


TMAX = length(Dlist);
%% create the adjacency matrix for all nearby detections
disp('...generate the graph');
tic
Ndetection = count-1;
A = make_adjacency(Dlist, D, WIN_SIZE, SPATIAL_DIST_THRESH, Ndetection);

% fill out all the distances in the adjacency matrix
edges = find(A == 1);
W = edges;
parfor i =1:length(edges)
    [r,c] = ind2sub(size(A), edges(i));
    W(i) = trkDetectionDistance(D(r), D(c), WT, WSH);%#ok
end
WW = A;
WW(edges) = W;
W = WW;
dt = toc;
disp(['...elapsed time for generating the graph is ' num2str(dt)])

%% apply the greedy tracking algorithm to link detections
disp('...greedy tracking');
tic
T = trkGreedyConnect2(W,A,D,W_THRESH);


%% get the track labels from T assigned to each detection
disp('...graph coloring');
[T tracks] = trkGraphColoring(T, MIN_TRACK_LENGTH); %#ok<*ASGLU>


%% assign ID's to each detections
disp('...assign ID''s to each detection')
for t = 1:TMAX
    for d = 1:length(Dlist{t}) % loop through detections in this time step
        detect_ind = Dlist{t}(d);
        D(detect_ind).ID = tracks(detect_ind); %#ok<*AGROW>
    end
end


%% get a list of detections and associated time steps for each track
[trkSeq, timeSeq] = getTrackSequences(Dlist, tracks, D);
dt = toc;
disp(['...elapsed time for tracking is ' num2str(dt)])