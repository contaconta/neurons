%% detect somata
function Soma = trkDetectSomataGrouped(M, J)




% dmax = length(D);
TMAX = length(M);

h = [1;1];
multFactor = 1.5;
Soma = cell(size(J));

parfor t = 1:TMAX
    mean_std = zeros(2, max(M{t}(:)));
    for i=1:max(M{t}(:))
       mean_std(1, i) = mean(J{t}(M{t} == i));
       mean_std(2, i) = std(J{t}(M{t} == i));
    end
    [U, V, L] = RegionGrowingSomata(h, J{t}, M{t}, mean_std, multFactor);
    SomaM  	= imfill((U < 0.000003) & (L < 7), 'holes');
    V(~SomaM) = 0;
    Soma{t} = V;
    
end