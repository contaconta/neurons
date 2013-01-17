function [Filaments, Regions, U, L] = trkDetectFilamentsGlobalThresh(Somata, Tubularity, GEODESIC_DISTANCE_NEURITE_THRESH)

TMAX = length(Somata);

Filaments = cell(size(Somata));

U = cell(size(Somata));
Regions = cell(size(Somata));
L = cell(size(Somata));

parfor t = 1:TMAX
    [UU, Regions{t}, L{t}] = RegionGrowingNeurites([1;1], Tubularity{t}, double(Somata{t}));
    Filaments{t} = bwmorph(UU < GEODESIC_DISTANCE_NEURITE_THRESH, 'skel', Inf);
    U{t} = UU;
end