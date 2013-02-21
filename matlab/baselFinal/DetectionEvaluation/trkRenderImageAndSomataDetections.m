function [Im, GT, Detections, TruePositives, FalsePositives, FalseNegatives] = trkRenderImageAndSomataDetections(Im, Cells, CellsList, AnnotatedTrackedCells, TruePositivesInput, FalsePositivesInput,  image_index)



% I = double(Im);
% I = 1- mat2gray(I);
% Ir = I; Ig = I; Ib = I;
Im = double(Im);
GT             = zeros(size(Im));

for i = 1:length(AnnotatedTrackedCells)
    for j = 1:numel(AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area)
        currentTime          = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.Time;
        if(currentTime == image_index)
            currentsomaGT     = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.PixelIdxList;
            GT(currentsomaGT) = 1;
        end
    end
end

% GT = cat(3, zeros(size(GT)), GT, zeros(size(GT)));
%%
Detections     = zeros(size(Im));

for j = CellsList{image_index}
    Detections(Cells(j).SomaPixelIdxList) = 1;
end

%%
TruePositives  = zeros(size(Im));
idxCell = 1;
for i = 1:length(AnnotatedTrackedCells)
    for j = 1:numel(AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area)
        currentTime          = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.Time;
        if(currentTime == image_index && TruePositivesInput(idxCell) > 0)
            currentsomaGT     = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.PixelIdxList;
            TruePositives(currentsomaGT) = 1;
        end
        idxCell = idxCell + 1;
    end
    
end

%%
FalseNegatives = zeros(size(Im));

idxCell = 1;
for i = 1:length(AnnotatedTrackedCells)
    for j = 1:numel(AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area)
        currentTime          = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.Time;
        if(currentTime == image_index && TruePositivesInput(idxCell) == 0)
            currentsomaGT     = AnnotatedTrackedCells{i}.soma.listOfObjects.t2_area{j}.PixelIdxList;
            FalseNegatives(currentsomaGT) = 1;
        end
        idxCell = idxCell + 1;
    end
    
end

%%
FalsePositives = zeros(size(Im));
for j = 1:length(FalsePositivesInput{image_index})
    if FalsePositivesInput{image_index}(j) == 0
        FalsePositives(Cells(CellsList{image_index}(j)).SomaPixelIdxList) = 1;
    end
end

