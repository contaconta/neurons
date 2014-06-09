% extract series of track labels and time stamps for each valid track
function [trkSeq, timeSeq] = trkGetTrackSequences(Dlist, tracks, D)
trkSeq = cell(1, max(tracks(:)));
timeSeq = cell(1, max(tracks(:)));
for i = 1:max(tracks(:))

    for t = 1:length(Dlist)
        detections = Dlist{t};
        ids = [D(detections).ID];

        d = detections(find(ids == i,1));

        if ~isempty(d)
            trkSeq{i} = [trkSeq{i} d];
            timeSeq{i} = [timeSeq{i} t];
        end
    end
end
