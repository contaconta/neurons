function [detectedGT, FalsePositives] = trkEvaluateDetections(Cells, CellsList, AnnotatedTrackedCells, overlappingTolerance)


NbGtCells = 0;
for i = 1:length(AnnotatedTrackedCells)
    NbGtCells = NbGtCells + AnnotatedTrackedCells{i}.LifeTime;
end

detectedGT = zeros(1, NbGtCells);
FalsePositives = CellsList;
for i =1:length(FalsePositives)
    FalsePositives{i} = 0*FalsePositives{i};
end

idxCell = 1;
for i = 1:length(AnnotatedTrackedCells)
    for j = 1:numel(AnnotatedTrackedCells{i}.nucleus.listOfObjects.t2_area)
        currentTime         = AnnotatedTrackedCells{i}.nucleus.listOfObjects.t2_area{j}.Time;
        currentNucleusGT    = AnnotatedTrackedCells{i}.nucleus.listOfObjects.t2_area{j}.PixelIdxList;
        inc = 1;
        for k = CellsList{currentTime}
           currentDetectedCell    = Cells(k);
           currentDetectedNucleus = currentDetectedCell.NucleusPixelIdxList;
           overlapping = numel(intersect(currentDetectedNucleus, currentNucleusGT)) / min( numel( currentNucleusGT ), numel(currentDetectedNucleus) );
           if overlapping > overlappingTolerance
               detectedGT(idxCell) = k;
               FalsePositives{currentTime}(inc) = 1;
           end
           inc = inc + 1;
        end
        idxCell = idxCell + 1;
    end
end
