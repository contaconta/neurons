function [SomataTracked] = trkGetTrackedSomata(Cells, Somata)

SomataTracked = Somata;
parfor t = 1:length(SomataTracked)
   SomataTracked{t} = zeros(size(Somata{t}));
end

for i = 1:length(Cells)
    if Cells(i).ID > 0
        tt = Cells(i).Time;
        SomataTracked{tt}(Cells(i).SomaPixelIdxList) = Cells(i).ID;
    end
end
