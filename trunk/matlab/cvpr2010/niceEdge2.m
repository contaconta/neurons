function E4 = niceEdge2(I)

CANNYSIGMA = 1;
CANNYTHRESH = 14;
MINAREA = 90;


E = canny(I, CANNYSIGMA);

E2 = E > CANNYTHRESH;

%E = canny(I, 1); E2 = E > 14;  imagesc(bwmorph(E2, 'erode'));

E2 = bwmorph(E2, 'erode');

%imagesc(E2);
%keyboard;
%L = bwlabel(E > CANNYTHRESH);
%STATS = regionprops(logical(E2), 'PixelIdxList', 'Area');
STATS = regionprops(bwlabel(logical(E2)), 'PixelIdxList', 'Area');

for l = 1:length(STATS)
if STATS(l).Area < MINAREA;
E2(STATS(l).PixelIdxList) = 0;
%disp(['killed region ' num2str(l)]);
end
end

%keyboard;

E3 = bwmorph(E2, 'thin', Inf);
E4 = bwmorph(E3, 'diag');
% add edges on the frames
E4(1:size(E4,1),1) = 1;
E4(1:size(E4,1),2) = 1;
E4(1:size(E4,1),size(E4,2)) = 1;
E4(1:size(E4,1),size(E4,2)-1) = 1;

E4(1,1:size(E4,2)) = 1;
E4(2,1:size(E4,2)) = 1;
E4(size(E4,1),  1:size(E4,2)) = 1;
E4(size(E4,1)-1,1:size(E4,2)) = 1;

%E4 = E2;