function [Regions, U, L] = trkDetectFilamentsGlobalThresh(Somata, Tubularity)

% Given the somata as starting points and the tubularity measures
% compute:
% 1- U       : The Geodesic distance map
% 2- Regions : The Voronoi partition:     
% 3- L       : The euclidian lengths of the geodesic paths
%
% authors: F. Benmansour

TMAX = length(Somata);

U = cell(size(Somata));
Regions = cell(size(Somata));
L = cell(size(Somata));

parfor t = 1:TMAX
    [UU, Regions{t}, L{t}] = RegionGrowingNeurites([1;1], Tubularity{t}, double(Somata{t}));
    U{t} = UU;
end