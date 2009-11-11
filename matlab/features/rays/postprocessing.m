function M = postprocessing(M, I)

M = bwmorph(M, 'majority');
M = bwmorph(M, 'fill', Inf);

MINSIZE = 75;

L = bwlabel(M);

SMOOTHINGWINDOW = 30;  %25
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

figure(2); imshow(I); hold on;
for k = 1:length(G)
    boundary = G{k};
    plot(boundary(:,2), boundary(:,1), 'Color', [0 .8 0], 'LineWidth', 2.5);
    plot(boundary(:,2), boundary(:,1), 'Color', [.3 .9 .3], 'LineStyle', '-', 'LineWidth', 1);
end

hold off;

%keyboard;