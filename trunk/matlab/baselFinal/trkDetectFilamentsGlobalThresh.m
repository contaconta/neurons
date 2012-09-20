function [Filaments, Regions, U, L] = trkDetectFilamentsGlobalThresh(Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH)

TMAX = length(Somata);

Filaments = cell(size(Somata));

U = cell(size(Somata));
Regions = cell(size(Somata));
L = cell(size(Somata));

parfor t = 1:TMAX
    [U{t}, Regions{t}, L{t}] = RegionGrowingNeurites([1;1], Tubularity{t}, double(Somata{t}));
    Filaments{t} = bwmorph(U{t} < GEODESIC_DISTANCE_NEURITE_THRESH, 'skel', Inf);
end