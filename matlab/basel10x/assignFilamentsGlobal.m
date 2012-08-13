function [FIL V L] = assignFilamentsGlobal(S, Tubularity)


FIL = cell(size(S)); 
V = cell(size(S));
L = cell(size(S));
TMAX = length(S);
parfor t = 1:TMAX
    [FIL{t}, V{t}, L{t}] = RegionGrowingNeurites([1;1], Tubularity{t}, double(S{t}));
end