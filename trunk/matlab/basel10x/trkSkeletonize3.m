function FILAMENTS = trkSkeletonize3(FIL, S, L, NEURITES_THRESHOLD)

FILAMENTS = cell(size(FIL));

parfor t = 1:length(FIL)
    UU = FIL{t};
    UU(UU==0) = 1;
    RR = L{t}./UU;
    FILAMENTS{t} = bwmorph(RR > NEURITES_THRESHOLD | S{t}, 'skel', Inf);
end