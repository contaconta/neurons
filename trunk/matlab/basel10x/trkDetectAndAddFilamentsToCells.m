function [Cells] = trkDetectAndAddFilamentsToCells(Cells, Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH)

[SomataTracked] = trkGetTrackedSomata(Cells, Somata);

[Filaments, Regions, U, Length] = trkDetectFilamentsGlobalThresh(SomataTracked, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH);

for i =1:length(Cells)
    if Cells(i).ID > 0
        t = Cells(i).Time;
        Cells(i).Neurites       = (Filaments{t} & (Regions{t} == Cells(i).ID) & (~SomataTracked{t}));
        Cells(i).NeuritesNumber = 0;
        Cells(i).NeuritesList   = [];
        
        Cells(i).TotalNeuritesLength     = 0;
        Cells(i).TotalNeuritesBranches  = 0;
        Cells(i).MeanNeuritesComplexity = nan;
    end
end

minimalSizeOfNeurite = 8;
NeuriteConnectivity  = 8;
RECURSIONLIMIT = 5000;
set(0,'RecursionLimit',RECURSIONLIMIT);

pad = 5;

%%
parfor dd = 1:length(Cells)
    if Cells(dd).ID > 0
        t = Cells(dd).Time;
        RR = (U{t} < GEODESIC_DISTANCE_NEURITE_THRESH)& (U{t} > 0) &  (Regions{t} == Cells(dd).ID);%#ok
        [B,~,~,A] = bwboundaries(RR, 'noholes');
        
        listOfCandidateEndPoints = [];
        for k=1:length(B),
            if(~sum(A(k,:)))
                boundary = B{k};
                Idx = sub2ind(size(U{t}), boundary(:, 1), boundary(:, 2));
                if(length(Idx) > pad)
                    Idx(end+1:end+pad) = Idx(1:pad);
                    LL = Length{t}(Idx);%#ok
                    LL = smooth(LL, pad);
                    if(max(LL) > pad)
                        [~,imax] = findpeaks( LL, 'MINPEAKHEIGHT', pad, 'MINPEAKDISTANCE', min(pad, floor(length(Idx)/2)));
                        Idx = Idx(1+pad:end);
                        imax(imax <=pad) = [];
                        listOfCandidateEndPoints = vertcat(listOfCandidateEndPoints, Idx(imax-pad));%ok
                    end
                end
            end
        end
        neurites = Cells(dd).Neurites;
        listOfCandidateEndPoints = intersect(find(bwmorph(neurites, 'dilate', 1)), listOfCandidateEndPoints);
        if(~isempty(listOfCandidateEndPoints))
            [r c] =ind2sub(size(neurites), listOfCandidateEndPoints);
            UU = U{t};
            UU(  Regions{t} ~= Cells(dd).ID ) = 1e9;% to garentee that the back propagation searches only in the region of interest
            neurites = BackPropagate([r c]', UU) ;
        else
            neurites = false(size(neurites));
        end
        neurites = bwmorph(bwmorph(neurites, 'dilate', 1), 'skel', Inf);
        neurites  = neurites & (Regions{t} == Cells(dd).ID);
    
        [LL, numberOfNeurites] = bwlabel(neurites, NeuriteConnectivity);
        for j = 1:numberOfNeurites
           if(sum(sum(LL == j)) < minimalSizeOfNeurite) 
               neurites(LL == j) = 0;
           end
        end
        Cells(dd).Neurites = neurites;
        [LL, numberOfNeurites] = bwlabel(neurites, NeuriteConnectivity);
        listOfNeurites = cell(1, numberOfNeurites);
        filam   = cell(1, numberOfNeurites);
        TotalNeuritesLength      = 0;
        TotalNeuritesBranches    = 0;
        listOfNeuritesComplexity = zeros(1, numberOfNeurites);
        for j =1:numberOfNeurites
            set(0,'RecursionLimit',RECURSIONLIMIT);
            listOfNeurites{j} = find(LL==j);
            [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites(Tubularity{Cells(dd).Time}, ...
                                                                    Cells(dd).SomaPixelIdxList, ...
                                                                    Cells(dd).NucleusCentroid,  ...
                                                                    listOfNeurites{j});%#ok
            filam{j}.Parents             = parents;
            filam{j}.NumKids             = branchesLeafs;
            filam{j}.NeuriteId           = neuriteId;
            filam{j}.NeuritePixelIdxList = listOfNeurites{j};
            filam{j}.IMSIZE              = size(neurites);
            filam{j}                     = trkFindBranches(filam{j});
            filam{j}                     = rmfield(filam{j}, 'IMSIZE');
            neuriteComplexity = nan;%#ok
            if filam{j}.TotalCableLength > 0
                neuriteComplexity  =   length(filam{j}.BranchLengthsDistribution) / filam{j}.TotalCableLength;
                filam{j}.NeuriteComplexity = neuriteComplexity;
            else
                error('a tree stored at this level should not be empty');
            end
            
            listOfNeuritesComplexity(j) = neuriteComplexity;
            
            TotalNeuritesLength          = TotalNeuritesLength + filam{j}.TotalCableLength;
            TotalNeuritesBranches        = TotalNeuritesBranches + length(filam{j}.BranchLengthsDistribution); 
        end
                
        Cells(dd).NeuritesNumber         = numberOfNeurites;
        Cells(dd).NeuritesList           = filam;
        Cells(dd).TotalNeuritesLength    = TotalNeuritesLength;
        Cells(dd).TotalNeuritesBranches  = TotalNeuritesBranches;
        Cells(dd).MeanNeuritesComplexity = mean(listOfNeuritesComplexity);
    end
end

% parfor dd = 1:length(Cells)
%     if Cells(dd).ID > 0
%         neurites = Cells(dd).Neurites;
%         [LL, numberOfNeurites] = bwlabel(neurites, NeuriteConnectivity);
%         for j = 1:numberOfNeurites
%            if(sum(sum(LL == j)) < minimalSizeOfNeurite) 
%                neurites(LL == j) = 0;
%            end
%         end
%         Cells(dd).Neurites = neurites;
%         [LL, numberOfNeurites] = bwlabel(neurites, NeuriteConnectivity);
%         listOfNeurites = cell(1, numberOfNeurites);
%         filam   = cell(1, numberOfNeurites);
%         for j =1:numberOfNeurites
%             set(0,'RecursionLimit',RECURSIONLIMIT);
%             listOfNeurites{j} = find(LL==j);
%             [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites(Tubularity{Cells(dd).Time}, ...
%                                                                     Cells(dd).SomaPixelIdxList, ...
%                                                                     Cells(dd).NucleusCentroid,  ...
%                                                                     listOfNeurites{j});%#ok
%             filam{j}.Parents             = parents;
%             filam{j}.NeuriteID           = neuriteId;
%             filam{j}.NumKids             = branchesLeafs;
%             filam{j}.NeuritePixelIdxList = listOfNeurites{j};
%         end
%         Cells(dd).NeuritesNumber            = numberOfNeurites;
%         Cells(dd).NeuritesList              = filam;
%     end
% end