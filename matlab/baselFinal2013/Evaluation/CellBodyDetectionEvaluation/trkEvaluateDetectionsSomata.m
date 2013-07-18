function [detectedGT, FalsePositives, detectedGTTracksId, detectedGTTime] = ...
    trkEvaluateDetectionsSomata(Cells, CellsList, AnnotatedTrackedCells, overlappingTolerance)


NbGtCells = 0;
for i = 1:length(AnnotatedTrackedCells)
    NbGtCells = NbGtCells + AnnotatedTrackedCells{i}.LifeTime;
end

detectedGT          = zeros(1, NbGtCells);
detectedGTTracksId  = zeros(1, NbGtCells);
detectedGTTime      = zeros(1, NbGtCells);


FalsePositives = CellsList;
for i =1:length(FalsePositives)
    FalsePositives{i} = 0*FalsePositives{i};
end

idxCell = 1;
for i = 1:length(AnnotatedTrackedCells)
    for j = 1:numel(AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area)
        currentTime         = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.Time;
        currentSomaGT    = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.PixelIdxList;
        inc = 1;
        for k = CellsList{currentTime}
           currentDetectedCell    = Cells(k);
           currentDetectedSoma = currentDetectedCell.SomaPixelIdxList;
           overlapping = numel(intersect(currentDetectedSoma, currentSomaGT)) / min( numel( currentSomaGT ), numel(currentDetectedSoma) );
           if overlapping > overlappingTolerance
               detectedGT(idxCell)              = k;
               detectedGTTracksId(idxCell)      = i;
               detectedGTTime(idxCell)          = currentTime;
               FalsePositives{currentTime}(inc) = 1;
           end
           inc = inc + 1;
        end
        idxCell = idxCell + 1;
    end
end
