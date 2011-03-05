function [T tracks] = trkGraphColoring(T,MIN_TRACK_LENGTH)


T = T | T' ;
T = double(T);
tracks = zeros(1,size(T,1));


T(T == 1) = -1;
id = 1;
first = find(T == -1, 1);



while ~isempty(first)

    [r,c] = ind2sub(size(T), first);
    
    [T, tracks] = rec_color(T,r,c,id, tracks);
    
    id = id + 1;
    first = find(T == -1, 1);
end


T = tril(T);

% get rid of short tracks
for i = 1:max(tracks)
    num_detects = sum(tracks == i);
    if num_detects < MIN_TRACK_LENGTH
        tracks(tracks == i) = 0;
    end
end



%keyboard;





%==================================================
function [T, tracks]  = rec_color(T,r,c,id,tracks)

T(r,c) = id;
T(c,r) = id;
tracks(r) = id;
tracks(c) = id;

Ninds = find(T(:,c) == -1)';
for i = 1:length(Ninds)
    [T, tracks] = rec_color(T, Ninds(i), c, id, tracks);
end

Ninds = find(T(:,r) == -1)';
for i = 1:length(Ninds)
    [T,tracks] = rec_color(T,r, Ninds(i), id,tracks);
end





