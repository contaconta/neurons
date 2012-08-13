function [Cells] = trkDetectAndAddFilamentsToCells(Cells, Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH)

[SomataTracked] = trkGetTrackedSomata(Cells, Somata);

[Filaments, Regions] = trkDetectFilamentsGlobalThresh(SomataTracked, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH);

for i =1:length(Cells)
    if Cells(i).ID > 0
        t = Cells(i).Time;
        Cells(i).Neurites       = (Filaments{t} & (Regions{t} == Cells(i).ID) & (~SomataTracked{t}));
        Cells(i).NeuritesNumber = 0;
        Cells(i).NeuritesList   = [];
    end
end

minimalSizeOfNeurite = 8;
NeuriteConnectivity  = 8;
RECURSIONLIMIT = 5000;
set(0,'RecursionLimit',RECURSIONLIMIT);


parfor dd = 1:length(Cells)
    if Cells(dd).ID > 0
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
end