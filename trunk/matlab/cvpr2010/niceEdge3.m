function E4 = niceEdge3(I, varargin)

CANNYSIGMA = 2;
CANNYTHRESH = 20;
%MINAREA = 60;
THRESH = 20;

if nargin == 1
    MINAREA = 60;
else
    MINAREA = varargin{1};
end

[E O] = canny(I, CANNYSIGMA);
N = nonmaxsup(E,O,4);
E2 = bwmorph( N > THRESH, 'dilate', 1);


STATS = regionprops(bwlabel(logical(E2)), 'PixelIdxList', 'Area'); %#ok<MRPBW>

for l = 1:length(STATS)
    if STATS(l).Area < MINAREA;
        E2(STATS(l).PixelIdxList) = 0;
    end
end

% E2 = E > CANNYTHRESH;
% 
% %E = canny(I, 1); E2 = E > 14;  imagesc(bwmorph(E2, 'erode'));
% 
% E2 = bwmorph(E2, 'erode');
% 
% %imagesc(E2);
% %keyboard;
% %L = bwlabel(E > CANNYTHRESH);
% %STATS = regionprops(logical(E2), 'PixelIdxList', 'Area');
% STATS = regionprops(bwlabel(logical(E2)), 'PixelIdxList', 'Area');
% 
% for l = 1:length(STATS)
% if STATS(l).Area < MINAREA;
% E2(STATS(l).PixelIdxList) = 0;
% %disp(['killed region ' num2str(l)]);
% end
% end
% 
% %keyboard;
% 
% % E3 = bwmorph(E2, 'thin', Inf);
% % E4 = bwmorph(E3, 'diag');

E4 = E2;
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