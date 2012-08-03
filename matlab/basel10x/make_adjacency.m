% create and adjacency matrix linking nearby detections
function A = make_adjacency(Dlist, D, WIN_SIZE, SPATIAL_DIST_THRESH, Ndetection)
A = zeros(Ndetection);
for t = 2:length(Dlist)
    for d = 1:length(Dlist{t})
        d_i = Dlist{t}(d);
        min_t = max(1, t-WIN_SIZE);
        for p = min_t:t-1
            d1 = D(d_i);
            for k = 1:length(Dlist{p})
                d2 = D(Dlist{p}(k));
                space_d = sqrt( (d1.Centroid(1) - d2.Centroid(1))^2 + (d1.Centroid(2) - d2.Centroid(2))^2);
                if(space_d < SPATIAL_DIST_THRESH)
                    A(d_i, Dlist{p}(k)) = 1;
                end
            end
        end
    end
end
