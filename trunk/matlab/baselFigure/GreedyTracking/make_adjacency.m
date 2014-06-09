function A = make_adjacency(Cells, CellsList, TEMPORAL_WIN_SIZE, SPATIAL_WINDOWS_SIZE, Ndetection)

A = zeros(Ndetection);
for t = 2:length(CellsList)
    for d = 1:length(CellsList{t})
        d_i = CellsList{t}(d);
        centroid_i = Cells(d_i).NucleusCentroid;
        min_t = max(1, t-TEMPORAL_WIN_SIZE);
        for p = min_t:t-1
            d_j = CellsList{p};
            for k = 1:length(d_j)
                centroid_j = Cells(d_j(k)).NucleusCentroid;
                dist = distance(centroid_i, centroid_j);
                if dist < SPATIAL_WINDOWS_SIZE
                    A(d_i, d_j(k)) = 1;
                end
            end
        end
    end
end