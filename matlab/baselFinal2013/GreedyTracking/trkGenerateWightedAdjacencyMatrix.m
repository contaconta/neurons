function [A, W] = trkGenerateWightedAdjacencyMatrix(Cells, CellsList, TEMPORAL_WIN_SIZE, SPATIAL_WINDOWS_SIZE, WT, WSH)

Ndetection = numel(Cells);
A = zeros(Ndetection);
W = zeros(Ndetection);
for t = 2:length(CellsList)
    for d = 1:length(CellsList{t})
        d_i = CellsList{t}(d);
        centroid_i = Cells(d_i).NucleusCentroid;
        min_t = max(1, t-TEMPORAL_WIN_SIZE);
        for p = min_t:t-1
            d_j = CellsList{p};
            for k = 1:length(d_j)
                centroid_j = Cells(d_j(k)).NucleusCentroid;
                dist = norm(centroid_i - centroid_j);%sqrt( (centroid_i(1) - centroid_j(1))^2 + (centroid_i(2) - centroid_j(2))^2 );
                if dist < SPATIAL_WINDOWS_SIZE
                    A(d_i, d_j(k)) = 1;
                    W(d_i, d_j(k)) = trkDetectionDistance(Cells(d_i), Cells(d_j(k)), WT, WSH);
                end
            end
        end
    end
end