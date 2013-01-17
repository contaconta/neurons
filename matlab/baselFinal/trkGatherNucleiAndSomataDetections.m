function [Cells CellsList] = trkGatherNucleiAndSomataDetections(Green, Red, Nuclei, Somata)

TMAX  = length(Nuclei);
Cells = [];
CellsList = cell(size(Nuclei));
count = 1;
minRed   =  1e9;
maxRed   = -1e9;
minGreen =  1e9;
maxGreen = -1e9;

for t = 1:TMAX
    detections_n = regionprops(Nuclei{t}, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    detections_s = regionprops(Somata{t}, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    if length(detections_n) ~= length(detections_s)
       error('the number of detected nuclei and somata should be the same !!') ;
    end
    
    minRed      = min(minRed,   min(Red{t}(:)));
    maxRed      = max(maxRed,   max(Red{t}(:)));
    minGreen    = min(minGreen, min(Green{t}(:)));
    maxGreen    = max(maxGreen, max(Green{t}(:)));
    
    if ~isempty(detections_n)
        for i =1:length(detections_n)
            % todo: improve this condition
            condition_to_keep_detection = detections_n(i).Eccentricity < 0.90;
            % todo: improve this condition
            if condition_to_keep_detection
                currentCell = [];
                currentCell.Time = t;
                % copy data for the neurite
                currentCell.NucleusArea               = detections_n(i).Area;
                currentCell.NucleusCentroid           = detections_n(i).Centroid;
                currentCell.NucleusEccentricity       = detections_n(i).Eccentricity;
                currentCell.NucleusMajorAxisLength    = detections_n(i).MajorAxisLength;
                currentCell.NucleusMinorAxisLength    = detections_n(i).MinorAxisLength;
                currentCell.NucleusOrientation        = detections_n(i).Orientation;
                currentCell.NucleusPerimeter          = detections_n(i).Perimeter;
                currentCell.NucleusCircularity        = 4*pi*currentCell.NucleusArea / (currentCell.NucleusPerimeter)^2;
                currentCell.NucleusPixelIdxList       = detections_n(i).PixelIdxList;
                currentCell.NucleusRedIntensities     = Red{t}(currentCell.NucleusPixelIdxList);
                currentCell.NucleusMeanRedIntensity   = sum(Red{t}(detections_n(i).PixelIdxList))/detections_n(i).Area;
                currentCell.NucleusMeanGreenIntensity = sum(Green{t}(detections_n(i).PixelIdxList))/detections_n(i).Area;
                % copy data for the soma
                currentCell.SomaArea                  = detections_s(i).Area;
                currentCell.SomaCentroid              = detections_s(i).Centroid;
                currentCell.SomaEccentricity          = detections_s(i).Eccentricity;
                currentCell.SomaMajorAxisLength       = detections_s(i).MajorAxisLength;
                currentCell.SomaMinorAxisLength       = detections_s(i).MinorAxisLength;
                currentCell.SomaOrientation           = detections_s(i).Orientation;
                currentCell.SomaPerimeter             = detections_s(i).Perimeter;
                currentCell.SomaCircularity           = 4*pi*currentCell.SomaArea / (currentCell.SomaPerimeter)^2;
                currentCell.SomaPixelIdxList          = detections_s(i).PixelIdxList;
                currentCell.SomaGreenIntensities      = Green{t}(currentCell.SomaPixelIdxList);
                currentCell.SomaHistGreen             = [];% just so that the parfor works well afterwards
                currentCell.SomaMeanGreenIntensity    = sum(Green{t}(detections_s(i).PixelIdxList))/detections_s(i).Area;
                % for neurites, the main loop is done later in a faster
                % parfor loop
                
                % store the cells
                if count == 1
                    Cells = currentCell;
                else
                    Cells(count) = currentCell;%#ok
                end
                CellsList{t} = [CellsList{t} count];
                count  = count + 1;
            end
        end
        
    end
end

Cells(end).MinRed   = minRed;
Cells(end).MaxRed   = maxRed;
Cells(end).MinGreen = minGreen;
Cells(end).MaxGreen = maxGreen;