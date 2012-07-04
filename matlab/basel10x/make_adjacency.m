% create and adjacency matrix linking nearby detections
function A = make_adjacency(Dlist, WIN_SIZE, Ndetection)
A = zeros(Ndetection);
for t = 2:length(Dlist)
    for d = 1:length(Dlist{t})
        d_i = Dlist{t}(d);
        min_t = max(1, t-WIN_SIZE);
        for p = min_t:t-1
            A(d_i, Dlist{p}) = 1;
        end
    end
end
