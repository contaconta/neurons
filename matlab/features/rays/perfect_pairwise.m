resultname = 'heathrowPerfect';

raysName = 'heathrowEdge7';

histFolder = '/osshare/DropBox/Dropbox/aurelien/FeatureVectors/histogram/heathrow/';
steerableFolder = '/osshare/Work/neurons/matlab/features/rays/featurevectors/heathrowSteerable/';
featureFolder = ['./featurevectors/' raysName '/'];
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrowAnnotations/';
imgFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrow/';
adjacencyFolder =  '/osshare/DropBox/Dropbox/aurelien/airplanes/neighbors/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/pairwise/' resultname '/'];
if ~isdir(destinationFolder); mkdir(destinationFolder); end;


d = dir([annotationFolder '*.png']);


for f = 1:length(d)
    
    disp(['Processing  ' d(f).name]);
    
    fileRoot = regexp(d(f).name, '(\w*)[^\.]', 'match');
 	fileRoot = fileRoot{1};
    
    I = imread([imgFolder fileRoot '.jpg']);
    load([featureFolder fileRoot '.mat']); 
    
    %C = readLabel([boundaryFolder fileRoot '.label' ], [size(I,1) size(I,2)])';
    C = imread([annotationFolder fileRoot '.png' ]); C0 = C(:,:,3) < 200; C1 = C(:,:,1) > 200; C2 = (C(:,:,1) <200 & C(:,:,3) >200); C = zeros(size(C0)) + C1 + 2.*C2;
        
    % load the Adjacency
    load([adjacencyFolder fileRoot '.mat']);
    STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
    
    
    % bg, boundary, mito labels needed for labeling boundaries pairs
    labels = zeros(size(STATS));
    for l=1:length(STATS)
        labels(l) = mode( C(STATS(l).PixelIdxList) );
    end
    
    
    % make an adjacency matrix of edges we need to evaluate
    E = triu(A) - speye(size(A));
    [r c] = find(E ~= 0);

    LA = zeros(length(r),1);
    probs = zeros(length(r),2);
    
    SIGMA1 = 0.15;
    SIGMA2 = 0.1;

    for x = 1:length(r)
       if (labels(r(x)) ~= 0) && (labels(c(x)) == 0)
            LA(x) = 1;  % a boundary exists here
            F = 1+ SIGMA1*randn(1,1);  F(F >1) = 2 - F( F > 1);
            probs(x,1) = F;
            probs(x,2) = 1 - probs(x,1);
        elseif (labels(c(x)) ~= 0) && (labels(r(x)) == 0)
            LA(x) = 1;  % a boundary exists here
            F = 1+ SIGMA1*randn(1,1);  F(F >1) = 2 - F( F > 1);
            probs(x,1) = F;
            probs(x,2) = 1 - probs(x,1);
        else
            LA(x) = 0;
            F = 1 + SIGMA2*randn(1,1);  F(F >1) = 2 - F( F > 1);
            %F = .2*randn(1,1); F(F<0) = -F(F < 0);
            probs(x,2) = F;
            probs(x,1) = 1 - probs(x,2);
        end
    end
    
    
%      for x = 1:length(r)
%        if (labels(r(x)) == 1) && (labels(c(x)) == 0)
%             LA(x) = 1;  % a boundary exists here
%             F = 1+ SIGMA1*randn(1,1);  F(F >1) = 2 - F( F > 1);
%             probs(x,1) = F;
%             probs(x,2) = 1 - probs(x,1);
%         elseif (labels(c(x)) == 1) && (labels(r(x)) == 0)
%             LA(x) = 1;  % a boundary exists here
%             F = 1+ SIGMA1*randn(1,1);  F(F >1) = 2 - F( F > 1);
%             probs(x,1) = F;
%             probs(x,2) = 1 - probs(x,1);
%         else
%             LA(x) = 0;
%             F = 1 + SIGMA2*randn(1,1);  F(F >1) = 2 - F( F > 1);
%             %F = .2*randn(1,1); F(F<0) = -F(F < 0);
%             probs(x,2) = F;
%             probs(x,1) = 1 - probs(x,2);
%         end
%     end
    
    locs = zeros(length(superpixels), 2);
        for s = superpixels
            locs(s,:) = STATS(s).Centroid;
        end
    
    P = sparse([],[],[], size(A,1), size(A,2),0);
    for x = 1:length(r)
        P(r(x),c(x)) = probs(x,1);
    end
    P = max(P,P');
    
    THRESHL = .5;    THRESHH = .85;
    hold off; figure(1); cla; imshow(I); hold on;
    gplot2(P > THRESHL ,locs, 'y-');
    gplot2(P > THRESHH, locs, 'r-');
    print(gcf, '-dpng', '-r150', [destinationFolder fileRoot '.png']);
    drawnow;  pause(0.01);
    
    writePairwisePrediction(destinationFolder, [fileRoot '.txt'], r, c, probs, LA, [1 0]);
       
    
end