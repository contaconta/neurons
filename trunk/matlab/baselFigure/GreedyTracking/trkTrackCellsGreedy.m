function [Cells, tracks, trkSeq, timeSeq] = trkTrackCellsGreedy(CellsList, Cells, ...
                                                              TrackingParameters)

%% create the adjacency matrix for all nearby detections
Ndetection = numel(Cells);
TEMPORAL_WINDOWS_SIZE       = TrackingParameters.TEMPORAL_WINDOWS_SIZE;
SPATIAL_WINDOWS_SIZE        = TrackingParameters.SPATIAL_WINDOWS_SIZE;
WT                          = TrackingParameters.WT;
WSH                         = TrackingParameters.WSH;
W_THRESH                    = TrackingParameters.W_THRESH;
MIN_TRACK_LENGTH            = TrackingParameters.MIN_TRACK_LENGTH;
NB_BEST_TRACKS              = TrackingParameters.NB_BEST_TRACKS;
TMAX                        = length(CellsList);

%% create the adjacency matrix for all nearby detections

[A, W] = trkGenerateWightedAdjacencyMatrix(Cells, CellsList, TEMPORAL_WINDOWS_SIZE, SPATIAL_WINDOWS_SIZE, WT, WSH);


%% apply the greedy tracking algorithm to link detections
disp('...greedy tracking');
T = trkGreedyConnect2(W,A,Cells,W_THRESH);


%% get the track labels from T assigned to each detection
%disp('...graph coloring');
[T tracks] = trkGraphColoring(T, MIN_TRACK_LENGTH); %#ok<*ASGLU>

%% pruning the tracks and sorting them according to their MeanGreenIntensity summed over lifetime

for k = 1:max(tracks)
    if ( length(find(tracks == k)) < MIN_TRACK_LENGTH)
        tracks(tracks == k) = 0;
    end
end
utracks = unique(tracks);
utracks = setdiff(utracks, 0);
for i = 1:length(utracks)
    tracks(tracks == utracks(i)) = i;
end

scoreTracks = zeros(1, length(utracks));
for i=1:length(scoreTracks)
    
    list_idx = find(tracks == i);
    for j = 1:length(list_idx)
        scoreTracks(i) = scoreTracks(i) + Cells(list_idx(j)).NucleusMeanGreenIntensity * numel(Cells(list_idx(j)).NucleusPixelIdxList);
    end
end

[scoreTracks, II] = sort(scoreTracks, 'descend');
sortedTracks = zeros(size(tracks));
for i=1:min(NB_BEST_TRACKS, length(scoreTracks))
    sortedTracks(tracks == II(i)) = i;
end
tracks = sortedTracks; 
%% assign ID's to each detections
for t = 1:TMAX
    for d = 1:length(CellsList{t}) % loop through detections in this time step
        detect_ind = CellsList{t}(d);
        Cells(detect_ind).ID = tracks(detect_ind); %#ok<*AGROW>
    end
end


%% get a list of detections and associated time steps for each track
[trkSeq, timeSeq] = trkGetTrackSequences(CellsList, tracks, Cells);
