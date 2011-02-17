function K = determineKfromMask(B, Kmin, Kmax, Afactor)
% B(12,12) = 1;
% B(12,13) = 1;

Bwhite = B > 0;
Bblack = B < 0;
w = regionprops(Bwhite, 'Area');
b = regionprops(Bblack, 'Area');

P = [w; b];

% CCw = bwconncomp(Bwhite,4);
% Pw = CCw.PixelIdxList;
% CCb = bwconncomp(Bblack,4);
% Pb = CCb.PixelIdxList;

% P = [Pw Pb];
%Kmin = N;

atotal = 0;
k = zeros(length(P),1);
for i = 1:length(P)
%     areai = length(P{i});
    areai = P(i).Area;
    atotal = atotal + areai;
    
    k(i) = max(1,round(areai / Afactor));
    
end
    
K = max(Kmin, sum(k));
K = min(Kmax, K);
K = min(atotal, K);



%   keyboard;
    
    %Npix = numel(find(B ~= 0));






