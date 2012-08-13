function [FILAMENTS, Regions] = trkDetectFilamentsGlobalThresh(Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH)

TMAX = length(Somata);

FILAMENTS = cell(size(Somata));

% FIL = cell(size(S));
Regions = cell(size(Somata));
% L = cell(size(S));

parfor t = 1:TMAX
    [U, Regions{t}, L] = RegionGrowingNeurites([1;1], Tubularity{t}, double(Somata{t}));
    FILAMENTS{t} = bwmorph(U < GEODESIC_DISTANCE_NEURITE_THRESH, 'skel', Inf);
end