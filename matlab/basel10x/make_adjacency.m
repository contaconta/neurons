% create and adjacency matrix linking nearby detections
function A = make_adjacency(Cellslist, Cells, WIN_SIZE, SPATIAL_DIST_THRESH, Ndetection)
A = zeros(Ndetection);
for t = 2:length(Cellslist)
    for d = 1:length(Cellslist{t})
        d_i = Cellslist{t}(d);
        min_t = max(1, t-WIN_SIZE);
        for p = min_t:t-1
            d1 = Cells(d_i);
            for k = 1:length(Cellslist{p})
                d2 = Cells(Cellslist{p}(k));
                space_d = sqrt( (d1.NucleusCentroid(1) - d2.NucleusCentroid(1))^2 + (d1.NucleusCentroid(2) - d2.NucleusCentroid(2))^2);
                if(space_d < SPATIAL_DIST_THRESH)
                    A(d_i, Cellslist{p}(k)) = 1;
                end
            end
        end
    end
end
