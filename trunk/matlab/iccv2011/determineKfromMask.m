function K = determineKfromMask(B, Kmin, Kmax, Afactor)


Bwhite = B > 0;
Bblack = B < 0;
CCw = bwconncomp(Bwhite,4);
%N = CCw.NumObjects;
Pw = CCw.PixelIdxList;
CCb = bwconncomp(Bblack,4);
%N = CCw.NumObjects;
Pb = CCb.PixelIdxList;

P = [Pw Pb];
%Kmin = N;

atotal = 0;
for i = 1:length(P)
    areai = length(P{i});
    atotal = atotal + areai;
    
    k(i) = max(1,round(areai / Afactor));
    
end
    
K = max(Kmin, sum(k));
K = min(Kmax, K);
K = min(atotal, K);



%   keyboard;
    
    %Npix = numel(find(B ~= 0));






