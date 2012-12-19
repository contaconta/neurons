function Somata = trkDetectSomataGlobal(M, Green, GEODESIC_DISTANCE_THRESH, LENGTH_THRESH, STD_MULT_FACTOR)

TMAX = length(M);

h = [1;1];
Somata = cell(size(Green));

parfor t = 1:TMAX
    Im = double(Green{t});
    mean_std = zeros(2, max(M{t}(:)));
    for i=1:max(M{t}(:))
       mean_std(1, i) = mean(Im(M{t} == i));
       mean_std(2, i) = std(Im(M{t} == i));
    end
    meanGlobal = mean(Im(M{t} == 0));
    stdGlobal  = std(Im(M{t} == 0));
    [U, V, L] = RegionGrowingSomata(h, Im, M{t}, mean_std, STD_MULT_FACTOR, meanGlobal, stdGlobal, LENGTH_THRESH);
    SomaM  	= imfill((U < GEODESIC_DISTANCE_THRESH) & (L < LENGTH_THRESH), 'holes');
    V(~SomaM) = 0;
    Somata{t} = V;
end