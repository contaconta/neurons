function [Cells CellsList] = trkGatherDetections(Green, Red, Nuclei, Somata, Filaments, Regions, Tubularity)

TMAX  = length(Nuclei);
Cells = [];
CellsList = cell(size(Nuclei));
count = 1;

RECURSIONLIMIT = 5000;
set(0,'RecursionLimit',RECURSIONLIMIT);

for t = 1:TMAX
    detections_n = regionprops(Nuclei{t}, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    detections_s = regionprops(Somata{t}, 'Area', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');  %#ok<*MRPBW>
    if length(detections_n) ~= length(detections_s)
       error('the number of detected nuclei and somata should be the same !!') ;
    end
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
                currentCell.NucleusPixelIdxList       = detections_n(i).PixelIdxList;
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
                currentCell.SomaPixelIdxList          = detections_s(i).PixelIdxList;
                currentCell.SomaMeangreenIntensity    = sum(Red{t}(detections_s(i).PixelIdxList))/detections_s(i).Area;
                % copy data for the neurites
                regionIdx = Regions{t}(detections_s(i).PixelIdxList(1));
                currentCell.Neurites = (Filaments{t} & (Regions{t} == regionIdx) & (~Somata{t}));
                currentCell.NeuritesNumber = 0;
                currentCell.NeuritesList = [];
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

minimalSizeOfNeurite = 8;
NeuriteConnectivity  = 8;

parfor dd = 1:length(Cells)
    neurites = Cells(dd).Neurites;
    [L, numberOfNeurites] = bwlabel(neurites, NeuriteConnectivity);
    for j = 1:numberOfNeurites
       if(sum(sum(L == j)) < minimalSizeOfNeurite) 
           neurites(L == j) = 0;
       end
    end
    Cells(dd).Neurites = neurites;
    [L, numberOfNeurites] = bwlabel(neurites, NeuriteConnectivity);
    listOfNeurites = cell(1, numberOfNeurites);
    filam   = cell(1, numberOfNeurites);
    for j =1:numberOfNeurites
        set(0,'RecursionLimit',RECURSIONLIMIT);
        listOfNeurites{j} = find(L==j);
        [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites(Tubularity{Cells(dd).Time}, ...
                                                                Cells(dd).SomaPixelIdxList, ...
                                                                Cells(dd).NucleusCentroid,  ...
                                                                listOfNeurites{j});%#ok
        filam{j}.Parents             = parents;
        filam{j}.NeuriteID           = neuriteId;
        filam{j}.NumKids             = branchesLeafs;
        filam{j}.NeuritePixelIdxList = listOfNeurites{j};
    end
    Cells(dd).NeuritesNumber            = numberOfNeurites;
    Cells(dd).NeuritesList              = filam;
end