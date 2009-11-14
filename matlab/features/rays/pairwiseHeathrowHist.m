
resultname = 'heathrowHist';

raysName = 'heathrowEdge7';


featureFolder = ['./featurevectors/' raysName '/'];
histFolder = '/osshare/DropBox/Dropbox/aurelien/FeatureVectors/histogram/heathrow/';
steerableFolder = '/osshare/Work/neurons/matlab/features/rays/featurevectors/heathrowSteerable/';
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrowAnnotations/';
imgFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrow/';
adjacencyFolder =  '/osshare/DropBox/Dropbox/aurelien/airplanes/neighbors/';
%boundaryFolder ='/osshare/DropBox/Dropbox/aurelien/superpixels/annotations/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/pairwise/' resultname '/'];

if ~isdir(destinationFolder); mkdir(destinationFolder); end;

addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');


%----------------------------------------------------------------------
% 1-20 Hist1
% 21-40 Hist2
%D = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
D = [];
for x = 1:40
    D(size(D,1)+1,:) = [x x]; %#ok<SAGROW> % Intensity
end
DMAX = 40;
%----------------------------------------------------------------------



% k-folds parameters
imgs = 1:13;                % list of image indexes
%K = 5;                      % the # of folds in k-fold training
TRAIN_LENGTH = 7000;        % the total # of examples per class in training set
BOUNDARY_LABEL = 1;



for k = 1:13
    % determine our training and testing images for this k-fold
    if k == 1; k1 = 1; else; k1 = (k-1)*K +1; end; %#ok<NOSEM>
%     testImgs = imgs( k1:min(k1+5-1, max(imgs)));
%     trainImgs = setdiff(imgs, testImgs);
    testImgs = k;
    trainImgs = setdiff(imgs, testImgs);
    disp(['Testing: ' num2str(testImgs)]);
    disp(['Training: ' num2str(trainImgs)]);
    
    % number of samples per class (N +, N-)
    N = round( TRAIN_LENGTH / length(trainImgs));
    NPOS = round(.4*N);
    NNEG = round(.6*N);
    
    % intialize the training vectors
    TRAIN = [];
    TRAIN_L = [];
    
    % index of the feature data
    d = dir([featureFolder '*.mat']);
    
    %% create the training vector looping through training images
    for i = trainImgs
        disp(['loading ' d(i).name]);
        fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
        fileRoot = fileRoot{1};
        % load the RAY features and the labels
        load([featureFolder d(i).name]); 
        % load the Hist features
        [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
        H = full(H);
        [lab S] = libsvmread([steerableFolder fileRoot '_u0_all_feature_vectors']);
        S = full(S);
        % load the 3-class annotation
        %C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        C = imread([annotationFolder fileRoot '.png' ]); C0 = C(:,:,3) < 200; C1 = C(:,:,1) > 200; C2 = (C(:,:,1) <200 & C(:,:,3) >200); 
        C = zeros(size(C0)) + C1 + 2.*C2;
    
        % load the normal annotation
        Q = imread([annotationFolder fileRoot '.png']); QR = Q(:,:,1) > 200; QG = Q(:,:,2) > 200; QB = Q(:,:,3) > 200;
        %Q = (QR | QB ) .* ~QG ;
        Q = QG .* ~(QR | QB);
        % load the Adjacency
        load([adjacencyFolder fileRoot '.mat']);
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        
        
        labels = zeros(size(STATS));  bootstrap = zeros(size(STATS));
        for l=1:length(STATS)
            labels(l) = mode( C(STATS(l).PixelIdxList) );
            bootstrap(l) = mode(Q(STATS(l).PixelIdxList) );
        end

        disp(['generating ' num2str(NPOS + NNEG) ' pairwise samples for ' d(i).name]);
        
        featureVector = zeros(NPOS + NNEG, DMAX);
        
        %% construct the positive examples - fuselage or wing next to background
        bnd = find(labels > 0);
        plist = randsample(bnd, min(length(bnd),4*NPOS))';
        
        c = 1; pc = 1;
        %M = zeros(size(L));
        while (c < NPOS) && (pc < length(plist))
            p = plist(pc);
            pc = pc +1;
            % we must find an adjacent background
            neighbors = find(A(p,:));
            bgns = find(labels(neighbors)==0);
            if length(bgns) > 1
                bg = neighbors(randsample(bgns,1));
                %M(STATS(p).PixelIdxList) = 2;
               % M(STATS(bg).PixelIdxList) = 1;
            elseif length(bgns) == 1
                bg = neighbors(bgns);
                %M(STATS(p).PixelIdxList) = 2;
                %M(STATS(bg).PixelIdxList) = 1;
            else
                continue;
            end
            featureVector(c,:) = [H(p,:)  H(bg,:)];
                      
            c = c + 1;
        end
       
        %figure; imagesc(M);
        
        %keyboard;
        
        
        %% construct the negative examples - background,  boundary next 
        % to mitochondria, and mitochondria interior
        
        
        % split the samples between the two cases
        N1 = round(NNEG/4);
        % fill in the fuselage/fuselage negative examples
        bnd = find(labels == 1);
        plist = randsample(bnd, min(length(bnd),3*N1))';
        
%         M = zeros(size(L));  
        pc = 1;
        %for p = plist
      	while (c < NPOS+N1) && (pc < length(plist))
            p = plist(pc);
            pc = pc +1;

            % we must find a background adjacent to mitochondria
            neighbors = find(A(p,:));
            mts = find(labels(neighbors)==1);
            if length(mts) > 1
                mt = neighbors(randsample(mts,1));
%                 M(STATS(p).PixelIdxList) = 2;
%                 M(STATS(mt).PixelIdxList) = 1;
            elseif ~isempty(mts ==1)
                mt = neighbors(mts);
%                 M(STATS(p).PixelIdxList) = 2;
%                 M(STATS(mt).PixelIdxList) = 1;
            else
                %disp('could not find any mitochondria neighbors!');
                continue;
            end
            featureVector(c,:) = [H(p,:)  H(mt,:)];          
            c = c + 1;
        end
        
        
        N2 = round(NNEG/4);
        % fill in the wing/wing examples
        mt = find(labels == 2);
        nlist = randsample(mt, min(length(mt), 3*N2))';
        nc = 1; %M = zeros(size(L));  
        
        %for n = nlist
        while (c < NPOS+N1+N2) && (nc < length(nlist))
            n = nlist(nc);
            nc = nc +1;    
            neighbors = find(A(n,:));
            mts = find(labels(neighbors)==2);
            if length(mts) > 1
                mt = neighbors(randsample(mts,1));
%                 M(STATS(n).PixelIdxList) = 2;
%               	M(STATS(mt).PixelIdxList) = 1;
            elseif ~isempty(mts == 1)
                mt = neighbors(mts);
%               	M(STATS(n).PixelIdxList) = 2;
%               	M(STATS(mt).PixelIdxList) = 1;
            else 
                continue;
            end
            featureVector(c,:) = [H(n,:) H(mt,:)];
            c = c + 1;
        end
        
        
        
        N3 = round(NNEG/4);
        % fill in the bootstrap examples
        im = find(bootstrap == 1);
        nlist = randsample(im, min(length(im), 3*N3))';
        %M = zeros(size(L));  
        nc = 1;
        
        %for n = nlist
        while (c < NPOS+N1+N2+N3) && (nc < length(nlist))
            n = nlist(nc);
            nc = nc +1; 
            neighbors = find(A(n,:));
            if length(neighbors) > 1
                im = randsample(neighbors, 1);
%                 M(STATS(n).PixelIdxList) = 2;
%               	M(STATS(im).PixelIdxList) = 1;
            elseif length(neighbors) == 1
                im = neighbors(1);
%                 M(STATS(n).PixelIdxList) = 2;
%               	M(STATS(im).PixelIdxList) = 1;
            else
                continue;
            end
            featureVector(c,:) = [ H(n,:) H(im,:) ];
            c = c + 1;
        end
     
        
        % the remaining are normal bg samples
        N4 = NPOS+NNEG - c + 1;
        
        % fill in the background examples
        bg = find(labels == 0);
        nlist = randsample(bg, min(length(bg), N4))';
       % M = zeros(size(L));  
        
        for n = nlist
            neighbors = find(A(n,:));
            bgns = find(labels(neighbors)==0);
            if length(bgns) > 1
                bg = neighbors(randsample(bgns,1));
%                 M(STATS(n).PixelIdxList) = 2;
%                 M(STATS(bg).PixelIdxList) = 1;
            else
                bg = neighbors(bgns);
%                 M(STATS(n).PixelIdxList) = 2;
%                 M(STATS(bg).PixelIdxList) = 1;
            end
            featureVector(c,:) = [H(n,:) H(bg,:)];
            c = c + 1;
        end
        

        clear RAYFEATUREVECTOR H S;

        TRAIN = [TRAIN; featureVector;]; %#ok<AGROW>
        TRAIN_L = [TRAIN_L; BOUNDARY_LABEL*ones(NPOS,1); zeros(NNEG,1)]; %#ok<AGROW>
    end
    
    disp([' rescaling the training data']);
    % rescale the data
    T1 = TRAIN; limits = zeros(size(D));
    for x = 1:size(D,1)
        limits(x,:) = [min(min(TRAIN(:,D(x,1):D(x,2)))) max(max(TRAIN(:,D(x,1):D(x,2))))];
        TRAIN(:,D(x,1):D(x,2)) = mat2gray(TRAIN(:,D(x,1):D(x,2)), limits(x,:));
    end
    

    %% =========== select parameters for the SVM =========================
    disp('Selecting parameters for the SVM');
    bestcv = 0;  
    CMIN = -1; CMAX = 3;  GMIN = -4; GMAX = 1;
    %CMIN = -1; CMAX = 3;  GMIN = -3; GMAX = -3;
    for log2c = CMIN:CMAX,
      for log2g = GMIN:GMAX,
        cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g) ' -m 500'];
        cv = svmtrain(TRAIN_L, TRAIN, cmd);
        if (cv >= bestcv),
          bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
      end
    end
    
    
    %% ============= train the SVM =======================================
    disp('Training the best SVM');
    cmd = ['-b 1 -c ' num2str(bestc) ' -g ' num2str(bestg) ' -m 500'];
    model = svmtrain(TRAIN_L, TRAIN, cmd);
    
    %% save the SVM model
    save([destinationFolder 'svm_model' num2str(testImgs) '.mat'], 'model', 'limits', 'D');
    
    
    

    
    %% loop through the test images and do prediction
    %======================================================================
    for i = testImgs
        disp(['Preparing to predict for ' d(i).name]);
        fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
        fileRoot = fileRoot{1};
        % load the RAY features and the labels
        load([featureFolder d(i).name]); 
        % load the Hist features
        [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
        H = full(H);
        [lab S] = libsvmread([steerableFolder fileRoot '_u0_all_feature_vectors']);
        S = full(S);
        % load the 3-class annotation
        %C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        C = imread([annotationFolder fileRoot '.png' ]); C0 = C(:,:,3) < 200; C1 = C(:,:,1) > 200; C2 = (C(:,:,1) <200 & C(:,:,3) >200); C = zeros(size(C0)) + C1 + 2.*C2;
    
        % load the Adjacency
        load([adjacencyFolder fileRoot '.mat']);
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        % load the image
        I = imread([imgFolder fileRoot '.jpg']);  I = rgb2gray(I);
        
        % bg, boundary, mito labels needed for labeling boundaries pairs
        labels = zeros(size(STATS));
        for l=1:length(STATS)
            labels(l) = mode( C(STATS(l).PixelIdxList) );
        end
        
        % make an adjacency matrix of edges we need to evaluate
        E = triu(A) - speye(size(A));
        [r c] = find(E ~= 0);
        
        % we need 2 feature vectors because we must check both directions of the pairwise term
        featureVector1 = zeros(length(r), DMAX);
        featureVector2 = zeros(length(r), DMAX);
        L1 = zeros(length(r),1);
        L2 = zeros(length(r),1);
        
        for x = 1:length(r)
            featureVector1(x,:) = [H(r(x),:) H(c(x),:)];
            featureVector2(x,:) = [ H(c(x),:) H(r(x),:) ];
            if (labels(r(x)) == 1) && (labels(c(x)) == 0)
                L1(x) = 1;  % a boundary exists here
                L2(x) = 0;  % wrong direction!
            elseif (labels(c(x)) == 1) && (labels(r(x)) == 0)
                L2(x) = 1;  % a boundary exists here
                L1(x) = 0;  % wrong direction!
            else
                L1(x) = 0;
                L2(x) = 0;
            end
        end
        

        % normalize the data 
        disp('normalizing');
        for x = 1:size(D,1)
            featureVector1(:,D(x,1):D(x,2)) = mat2gray(featureVector1(:,D(x,1):D(x,2)), limits(x,:));
            featureVector2(:,D(x,1):D(x,2)) = mat2gray(featureVector2(:,D(x,1):D(x,2)), limits(x,:));
        end
    
        % perform the SVM prediction
        cmd = '-b 1';
        disp('predicting 1/2');
        [pre_L1, acc1, probs1] = svmpredict(L1, featureVector1, model, cmd); disp('predicting 2/2');
        [pre_L2, acc2, probs2] = svmpredict(L2, featureVector2, model, cmd);
    

        
        % determine centroid locations on adjacency graph
        locs = zeros(length(superpixels), 2);
        for s = superpixels
            locs(s,:) = STATS(s).Centroid;
        end

        
        
        % combine our 2 directional predictions
        ind = find(model.Label == 1);       
        probsCUT = max(probs1(:,ind), probs2(:,ind));
        probsNOT = [ 1 - probsCUT];
        if ind == 1
            probs = [probsCUT probsNOT];
        else
            probs = [probsNOT probsCUT];
        end
       
        
        P = sparse([],[],[], size(A,1), size(A,2),0);
        for x = 1:length(r)
            P(r(x),c(x)) = probs(x);
        end
        P = max(P,P');

        % create and write the image
        THRESHL = .5;    THRESHH = .85;
        hold off; figure(1); cla; imshow(I); hold on;
        gplot2(P > THRESHL ,locs, 'y-');
        gplot2(P > THRESHH, locs, 'r-');
        print(gcf, '-dpng', '-r150', [destinationFolder fileRoot '.png']);
        drawnow;  pause(0.01);
        
        % write predictions to a text file
     	predL = probsCUT > THRESHL;
        predH = probsCUT > THRESHH;
        writePairwisePrediction(destinationFolder, [fileRoot '.txt'], r, c, probs, predL, model.Label);
       
        
        
        % check against the ground truth
        gt = max(L1,L2);
        ACC = rocstats(predL, gt, 'ACC');
        disp([num2str(ACC*100) '% accuracy on ' fileRoot]);
        ACC2 = rocstats(predH, gt, 'ACC');
        disp([num2str(ACC2*100) '% accuracy on ' fileRoot '  using >' num2str(THRESHH) ' probability']);
        fid = fopen([destinationFolder 'results.txt'], 'a'); fprintf(fid, '%g accuracy on %s\n', ACC*100, fileRoot); fclose(fid);
        fid = fopen([destinationFolder 'results.txt'], 'a'); fprintf(fid, '%g accuracy on %s when using >%g probability\n', ACC2*100, fileRoot, THRESHH); fclose(fid);

        
    end
    %keyboard;
end