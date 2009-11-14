function M2 = postprocessing(M, I, Icoeff)

M = bwmorph(M, 'majority');
M = bwmorph(M, 'fill', Inf);

MINSIZE = 50;

L = bwlabel(M);

SMOOTHINGWINDOW = 25;  %25
LOWESSSPAN = .1;


STATS = regionprops(L, 'Area', 'PixelIdxList');

for l = 1:length(STATS)
    if STATS(l).Area < MINSIZE
        M(STATS(l).PixelIdxList) = 0;
    end
end

L = bwlabel(M);


[B,L] = bwboundaries(L,8, 'holes'); %#ok<NASGU>

M2 = zeros(size(M));

for b = 1:length(B)
    Bb = B{b,:};
    Bx = smooth(Bb(:,1), SMOOTHINGWINDOW);
    By = smooth(Bb(:,2), SMOOTHINGWINDOW);
    %Bx = smooth(Bb(:,1), LOWESSSPAN, 'loess');
    %By = smooth(Bb(:,2), LOWESSSPAN, 'loess');
    
    G{b} =[Bx By]; %#ok<NASGU>
    
    Mb = poly2mask(By, Bx, size(M,1), size(M,2));
    M2 = M2 + Mb;
end

M2 = M2 > 0;

if Icoeff ~= 1
    idx=find(L~=0);
    I2=I;        
    I2(idx)=I2(idx)*Icoeff;
    figure(2); imshow(I2); hold on;
else
    figure(2); imshow(I); hold on;
end

outercolor = [.2039 .2824 .5765];
innercolor = [.3529 .4902 1];

set(gca, 'Position', [0 0 1 1]);
for k = 1:length(G)
    boundary = G{k};
    plot(boundary(:,2), boundary(:,1), 'Color', outercolor, 'LineWidth', 2);
    plot(boundary(:,2), boundary(:,1), 'Color', innercolor, 'LineStyle', '-', 'LineWidth', .7);
end

hold off;

%keyboard;
