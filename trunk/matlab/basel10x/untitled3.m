FF = zeros(size(U));
for i =1:max(Regions{1}(:))
    UU = U;
    UU(U==0) = 1;
    RR = L./U;
    Thresh(i) = min(U(RR(V == i) == max(RR(V == i))), 1e-2);
    FF(V == i & U < Thresh(i)) = 1;
end 
%%
clc
tic
FF = zeros(size(U));
V = Regions{1};
clear Thresh


for i =2:2%max(Regions{1}(:))
    
    pixelList = U(V == i);
    IntList = double(G{1}(V==i));
    idx = 1;
    TAB = linspace(min(pixelList), max(pixelList), 1000);
    for thr =TAB
        KK(idx) = sum(pixelList <  thr);
        GG1 = IntList(pixelList < thr);
        GG2 = IntList(pixelList >= thr);
        OTSU(idx) = (sum(pixelList <  thr) / length(pixelList) )*(sum(abs(GG1-mean(GG1)) )^2 - sum((GG1-mean(GG1)).^2))...
                + (sum(pixelList >= thr) / length(pixelList) )*(sum(abs(GG2- mean(GG2)))^2 - sum((GG2-mean(GG2)).^2));
            
            
%         OTSU(idx) = ((sum(GG1)^2 - sum(GG1.^2))) / length(pixelList <  thr);
        idx = idx + 1;
    end
    [~, idx] = min(OTSU);
    Thresh(i) = TAB(idx);
    FF(V == i & U < Thresh(i)) = 1;
end 
toc
%%
 i =26
UU = zeros(size(U));
UU(V == i) = U(V ==i);
figure; imshow(UU, []); colormap(jet);
%% try MSER
idx = 23;
% UU = uint8(255*mat2gray(U));
Ulist = U(V == idx);
UU = U;
UU(V~=idx) = 0;
numberofPoints = 500;
eps = 1e-7;

clear v;
tab = linspace(min(Ulist)+eps, max(Ulist), numberofPoints);
delta = 1;
idx = 1;
T = tab(1:end-delta);
for i = 1:length(T)
   v(idx) =  sum(Ulist> tab(i) & Ulist <= tab(i+delta) ) /  sum(Ulist <=tab(i+delta));
   idx = idx+1;
end
figure; imshow(UU, []); colormap(jet)

v = smooth(v, sm);
figure; plot(tab(1:end-delta), v); hold on;

[ymax,imax,ymin,imin] = extrema( v );

plot(T(imax),ymax,'r*',T(imin),ymin,'g*')