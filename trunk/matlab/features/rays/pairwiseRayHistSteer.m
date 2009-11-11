
resultname = 'RayHistSteer';

raysFolderName = 'rays30MedianInvariantE2';

histFolder = '/osshare/DropBox/Dropbox/aurelien/FeatureVectors/histogram/';
featureFolder = ['./featurevectors/' raysFolderName '/'];
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
boundaryFolder = '/osshare/DropBox/Dropbox/aurelien/superpixels/annotations/';
imgFolder = '/osshare/Work/Data/LabelMe/Images/fibsem/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/pairwise/' resultname '/'];
adjacencyFolder =  '/osshare/DropBox/Dropbox/aurelien/superpixels/neighbors/';
steerableFolder = './featurevectors/steerable_featureVectors/';
if ~isdir(destinationFolder); mkdir(destinationFolder); end;

addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');


%----------------------------------------------------------------------
% 1 mean I
% 2 var I
% 3-14 ray1
% 15-26 ray3
% 27-38 ray4
% 39-104 ray2
% 105-124 Independant Hists
% 125-208 Steer
D = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
for x = 105:204
    D(size(D,1)+1,:) = [x x]; % Intensity & steer
end
D2 = D + 204;
D = [D; D2];
DMAX = 408;
%----------------------------------------------------------------------


% 
% %----------------------------------------------------------------------
% % 1 mean I
% % 2 var I
% % 3-14 ray1
% % 15-26 ray3
% % 27-38 ray4
% % 39-104 ray2
% % 105-124 Independant Hists
% D = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
% for x = 105:124
%     D(size(D,1)+1,:) = [x x]; % Intensity
% end
% D2 = D + 124;
% D = [D; D2];
% DMAX = 248;
% %----------------------------------------------------------------------




% k-folds parameters
imgs = 1:23;                % list of image indexes
K = 5;                      % the # of folds in k-fold training
TRAIN_LENGTH = 8000;        % the total # of examples per class in training set
BOUNDARY_LABEL = 1;



for k = 1:5
    % determine our training and testing images for this k-fold
    if k == 1; k1 = 1; else; k1 = (k-1)*K +1; end; %#ok<NOSEM>
    testImgs = imgs( k1:min(k1+5-1, max(imgs)));
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
        C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        % load the normal annotation
        Q = imread([annotationFolder fileRoot '.png']); QR = Q(:,:,1) > 200; QG = Q(:,:,2) > 200; QB = Q(:,:,3) > 200;
        Q = (QR | QB ) .* ~QG ;
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
        
        %% construct the positive examples - boundaries next to background
        bnd = find(labels == 1);
        plist = randsample(bnd, NPOS)';
        
        c = 1;
        for p = plist
            % we must find a background adjacent to the boundary
            neighbors = find(A(p,:));
            bgns = find(labels(neighbors)==0);
            if length(bgns) > 1
                bg = neighbors(randsample(bgns,1));
            else
                bg = neighbors(bgns);
            end
            featureVector(c,:) = [RAYFEATUREVECTOR(p,:) H(p,:) S(p,:) RAYFEATUREVECTOR(bg,:) H(bg,:) S(bg,:)];
                      
            c = c + 1;
        end
        
        
        %% construct the negative examples - background,  boundary next 
        % to mitochondria, and mitochondria interior
        
        % split the samples between the two cases
        N1 = round(NNEG/5);
        % fill in the boundary/mitochondria negative examples
        bnd = find(labels == 1);
        plist = randsample(bnd, N1)';
        
        for p = plist
            % we must find a background adjacent to mitochondria
            neighbors = find(A(p,:));
            mts = find(labels(neighbors)==2);
            if length(mts) > 1
                mt = neighbors(randsample(mts,1));
            elseif ~isempty(mts ==1)
                mt = neighbors(mts);
            else
                %disp('could not find any mitochondria neighbors!');
                continue;
            end
            featureVector(c,:) = [RAYFEATUREVECTOR(p,:) H(p,:) S(p,:) RAYFEATUREVECTOR(mt,:) H(mt,:) S(mt,:)];          
            c = c + 1;
        end
        
        N2 = round(NNEG/5);
        % fill in the mitochondria/mitochondria examples
        mt = find(labels == 2);
        nlist = randsample(mt, N2)';
        
        for n = nlist
            neighbors = find(A(n,:));
            mts = find(labels(neighbors)==2);
            if length(mts) > 1
                mt = neighbors(randsample(mts,1));
            elseif ~isempty(mts == 1)
                mt = neighbors(mts);
            else 
                continue;
            end
            featureVector(c,:) = [RAYFEATUREVECTOR(n,:) H(n,:) S(n,:) RAYFEATUREVECTOR(mt,:) H(mt,:) S(mt,:)];
            c = c + 1;
        end
        
        N3 = round(NNEG/5);
        % fill in the boundary/boundary examples
        bd = find(labels == 1);
        nlist = randsample(bd, N3)';
        
        for n = nlist
            neighbors = find(A(n,:));
            bds = find(labels(neighbors)==2);
            if length(bds) > 1
                bd = neighbors(randsample(bds,1));
            elseif ~isempty(bds == 1)
                bd = neighbors(bds);
            else 
                continue;
            end
            featureVector(c,:) = [RAYFEATUREVECTOR(n,:) H(n,:) S(n,:) RAYFEATUREVECTOR(bd,:) H(bd,:) S(bd,:)];
            c = c + 1;
        end
        
        N4 = round(NNEG/5);
        % fill in the blue examples
        im = find(bootstrap == 1);
        nlist = randsample(im, N4)';
        for n = nlist
            neighbors = find(A(n,:));
            if length(neighbors) > 1
                im = randsample(neighbors, 1);
            elseif length(neighbors) == 1
                im = neighbors(1);
            else
                continue;
            end
            featureVector(c,:) = [RAYFEATUREVECTOR(n,:) H(n,:) S(n,:) RAYFEATUREVECTOR(im,:) H(im,:) S(im,:)];
            c = c + 1;
        end
        
        
        % the remaining are normal bg samples
        N5 = NPOS+NNEG - c + 1;
        
        % fill in the background examples
        bg = find(labels == 0);
        nlist = randsample(bg, N5)';
        
        for n = nlist
            neighbors = find(A(n,:));
            bgns = find(labels(neighbors)==0);
            if length(bgns) > 1
                bg = neighbors(randsample(bgns,1));
            else
                bg = neighbors(bgns);
            end
            featureVector(c,:) = [RAYFEATUREVECTOR(n,:) H(n,:) S(n,:) RAYFEATUREVECTOR(bg,:) H(bg,:) S(bg,:)];
            c = c + 1;
        end
        
        clear RAYFEATUREVECTOR H;

        TRAIN = [TRAIN; featureVector;]; %#ok<AGROW>
        TRAIN_L = [TRAIN_L; BOUNDARY_LABEL*ones(NPOS,1); zeros(NNEG,1)]; %#ok<AGROW>
    end
    
    disp([' rescaling the data for ' d(i).name]);
    % rescale the data
    T1 = TRAIN; limits = zeros(size(D));
    for x = 1:size(D,1)
        limits(x,:) = [min(min(TRAIN(:,D(x,1):D(x,2)))) max(max(TRAIN(:,D(x,1):D(x,2))))];
        TRAIN(:,D(x,1):D(x,2)) = mat2gray(TRAIN(:,D(x,1):D(x,2)), limits(x,:));
    end
    

    %% =========== select parameters for the SVM =========================
    disp('Selecting parameters for the SVM');
    bestcv = 0;  
    %CMIN = -1; CMAX = 3;  GMIN = -4; GMAX = 1;
    CMIN = -1; CMAX = 3;  GMIN = -3; GMAX = -3;
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
        C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        % load the Adjacency
        load([adjacencyFolder fileRoot '.mat']);
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        % load the image
        I = imread([imgFolder fileRoot '.png']);
        
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
            featureVector1(x,:) = [RAYFEATUREVECTOR(r(x),:) H(r(x),:) S(r(x),:) RAYFEATUREVECTOR(c(x),:) H(c(x),:) S(c(x),:)];
            featureVector2(x,:) = [RAYFEATUREVECTOR(c(x),:) H(c(x),:) S(c(x),:) RAYFEATUREVECTOR(r(x),:) H(r(x),:) S(r(x),:)];
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
end