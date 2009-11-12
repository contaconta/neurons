function E4 = niceEdge5(I, L)

L1 = L >0;


CANNYSIGMA = 2;
%MINAREA = 60;
THRESH = 20;

% [E O] = canny(I, CANNYSIGMA);
% N = nonmaxsup(E,O,1.25);
% E2 = bwmorph( N > THRESH, 'dilate', 1);
% E2 = bwmorph( E2, 'erode', 1);

E2 = edge(I, 'canny', .12, 3);

E2 = E2.*(~L1);
E2 = E2+edge(L);

%E2 = edge(L);

E4 = E2 > 0;

% 
% STATS = regionprops(bwlabel(logical(E2)), 'PixelIdxList', 'Area'); %#ok<MRPBW>
% 
% for l = 1:length(STATS)
%     if STATS(l).Area < MINAREA;
%         E2(STATS(l).PixelIdxList) = 0;
%     end
% end


% add edges on the frames
E4(1:size(E4,1),1) = 1;
E4(1:size(E4,1),2) = 1;
E4(1:size(E4,1),size(E4,2)) = 1;
E4(1:size(E4,1),size(E4,2)-1) = 1;

E4(1,1:size(E4,2)) = 1;
E4(2,1:size(E4,2)) = 1;
E4(size(E4,1),  1:size(E4,2)) = 1;
E4(size(E4,1)-1,1:size(E4,2)) = 1;