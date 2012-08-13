function [FILAMENTS, Regions] = trkDetectFilamentsGlobal(Somata, Tubularity, NEURITES_COMPOSIT_THRESH, NMS_HWS)

TMAX = length(Somata);

FILAMENTS = cell(size(Somata));

% FIL = cell(size(S));
Regions = cell(size(Somata));
% L = cell(size(S));

parfor t = 1:TMAX
    [U, Regions{t}, L] = RegionGrowingNeurites([1;1], Tubularity{t}, double(Somata{t}));
    UU = U;
    UU(UU==0) = 1;
    RR = L./UU;
    [r, c] = nonmaxsuppts(RR, NMS_HWS, NEURITES_COMPOSIT_THRESH);
    FILAMENTS{t} = BackPropagate([r, c]', U);
end