% fileRoot = 'heathrow9';
% 
% imgFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrow/';
% annotationFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrowAnnotations/';
% raysName = 'heathrowEdge6';
% featureFolder = ['./featurevectors/' raysName '/'];
% load([featureFolder fileRoot '.mat']); 
% I = imread([imgFolder fileRoot '.jpg']);  Ig = rgb2gray(I);
% imshow(I);
% SUPERMASK = ~imseg(zeros(size(Ig)), L) > 0;



SUPERMASK2 = SUPERMASK.*~C  + C > 0;

IR = I(:,:,1);
IG = I(:,:,2);
IB = I(:,:,3);
IR(~SUPERMASK2) = 50;
IG(~SUPERMASK2) = 50;
IB(~SUPERMASK2) = 50;
Is(:,:,1) = IR;
Is(:,:,2) = IG;
Is(:,:,3) = IB;

% %Is = imseg(I,L);
% figure(1); imshow(I);
% 
% 
% 
% 
% 
% 
% imwrite(I, 'Fig3a.png', 'PNG');


% C = imread([annotationFolder fileRoot '.png' ]); C0 = C(:,:,3) < 200; C1 = C(:,:,1) > 200; C2 = (C(:,:,1) <200 & C(:,:,3) >200); C = zeros(size(C0)) + C1 + 2.*C2;
%     
% 
% 
% % load the Adjacency
% adjacencyFolder =  '/osshare/DropBox/Dropbox/aurelien/airplanes/neighbors/';
% load([adjacencyFolder fileRoot '.mat']);
% STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
% 
% 
% % bg, boundary, mito labels needed for labeling boundaries pairs
% labels = zeros(size(STATS));
% for l=1:length(STATS)
%     labels(l) = mode( C(STATS(l).PixelIdxList) );
% end
% 
% 
% % plot an example of superpixels belonging to the airplanes
% E = triu(A) - speye(size(A));
% [r c] = find(E ~= 0);
% LA = zeros(length(r),1);
% 
% for x = 1:length(r)
%     if (labels(r(x)) ~= 0) && (labels(c(x)) ~= 0)
%         LA(x) = 1;  % a boundary exists here
%     else
%         LA(x) = 0;
%     end
% end
% 
% locs = zeros(length(superpixels), 2);
% for s = superpixels
%     locs(s,:) = STATS(s).Centroid;
% end
% 
% P = sparse([],[],[], size(A,1), size(A,2),0);
% for x = 1:length(r)
%     P(r(x),c(x)) = LA(x);
% end
% P = max(P,P');
% 
% 
% hold off; figure(2); cla; imshow(I); hold on;
% %gplot2(P ,locs, 'b.-');
%gplot2(P ,locs, 'b.-', 'LineWidth', .5);
% %gplot2(P, locs, 'k-');
% axis([0 800 450 979]);
% print(gcf, '-dpng', '-r300', 'Fig3b.png');





% make an adjacency matrix of edges we need to evaluate
E = triu(A) - speye(size(A));
[r c] = find(E ~= 0);

LA = zeros(length(r),1);

for x = 1:length(r)
   if (labels(r(x)) ~= 0) && (labels(c(x)) == 0)
        LA(x) = 1;  % a boundary exists here
        
    elseif (labels(c(x)) ~= 0) && (labels(r(x)) == 0)
        LA(x) = 1;  % a boundary exists here
       
    else
        LA(x) = 0;
       
    end
end


CUT = sparse([],[],[], size(A,1), size(A,2),0);
for x = 1:length(r)
    CUT(r(x),c(x)) = LA(x);
end
CUT = max(CUT,CUT');
clf;
imshow(imoverlay(Is, C, 'color', [0 0 .8], 'alpha', .75));
hold off; figure(1); cla; imshow(Is); hold on;
gplot2(CUT ,locs, 'r.-', 'LineWidth', .5);
gplot2(P ,locs, 'b.-', 'LineWidth', .5);
axis([0 800 450 979]);







% % make an adjacency matrix of edges we need to evaluate
% E = triu(A) - speye(size(A));
% [r c] = find(E ~= 0);
% 
% LA = zeros(length(r),1);
% probs = zeros(length(r),2);
% 
% SIGMA1 = 0.15;
% SIGMA2 = 0.1;
% 
% for x = 1:length(r)
%    if (labels(r(x)) ~= 0) && (labels(c(x)) == 0)
%         LA(x) = 1;  % a boundary exists here
%         F = 1+ SIGMA1*randn(1,1);  F(F >1) = 2 - F( F > 1);
%         probs(x,1) = F;
%         probs(x,2) = 1 - probs(x,1);
%     elseif (labels(c(x)) ~= 0) && (labels(r(x)) == 0)
%         LA(x) = 1;  % a boundary exists here
%         F = 1+ SIGMA1*randn(1,1);  F(F >1) = 2 - F( F > 1);
%         probs(x,1) = F;
%         probs(x,2) = 1 - probs(x,1);
%     else
%         LA(x) = 0;
%         F = 1 + SIGMA2*randn(1,1);  F(F >1) = 2 - F( F > 1);
%         %F = .2*randn(1,1); F(F<0) = -F(F < 0);
%         probs(x,2) = F;
%         probs(x,1) = 1 - probs(x,2);
%     end
% end
% 
% 
% 
% 
% P = sparse([],[],[], size(A,1), size(A,2),0);
% for x = 1:length(r)
%     P(r(x),c(x)) = probs(x,1);
% end
% P = max(P,P');
% 
% THRESHL = .5;    THRESHH = .85;
% hold off; figure(1); cla; imshow(I); hold on;
% gplot2(P > THRESHL ,locs, 'y-');
% gplot2(P > THRESHH, locs, 'r-');
% print(gcf, '-dpng', '-r150', [destinationFolder fileRoot '.png']);
% drawnow;  pause(0.01);
% 
% writePairwisePrediction(destinationFolder, [fileRoot '.txt'], r, c, probs, LA, [1 0]);
