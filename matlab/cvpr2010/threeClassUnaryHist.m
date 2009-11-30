
resultname = '3classHist';

raysName = 'rays30MedianInvariantE2';

boundaryFolder = '/osshare/DropBox/Dropbox/aurelien/superpixels/annotations/';
histFolder = '/osshare/DropBox/Dropbox/aurelien/FeatureVectors/histogram/';
steerableFolder = './featurevectors/steerable_featureVectors/';
featureFolder = ['./featurevectors/' raysName '/'];
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
imgFolder = '/osshare/Work/Data/LabelMe/Images/fibsem/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/unary/' resultname '/'];
if ~isdir(destinationFolder); mkdir(destinationFolder); end;

addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');



% k-folds parameters
imgs = 1:23;                % list of image indexes
K = 5;                      % the # of folds in k-fold training
TRAIN_LENGTH = 6000;        % the total # of examples per class in training set
MITO_LABEL = 2;             % label used for mito
BND_LABEL = 1;              % boundary label

%----------------------------------------------------------------------
%D = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
D = [];
for x = 1:20
    D(size(D,1)+1,:) = [x x]; % Intensity
end
%----------------------------------------------------------------------


for k = 1:5
    % determine our training and testing images for this k-fold
    if k == 1; k1 = 1; else; k1 = (k-1)*K +1; end; %#ok<NOSEM>
    testImgs = imgs( k1:min(k1+5-1, max(imgs)));
    trainImgs = setdiff(imgs, testImgs);
    disp(['Testing: ' num2str(testImgs)]);
    disp(['Training: ' num2str(trainImgs)]);
    
    % number of samples per class (N +, N-)
    N = round( TRAIN_LENGTH / length(trainImgs));
    N2 = round(.33*N);
    N1 = round(.33*N);
    N0 = round(.33*N);
    
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
        labels = mito; clear mito;
        % load the Hist features
        [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
        H = full(H);
        % load the normal annotation
        Q = imread([annotationFolder fileRoot '.png']); QR = Q(:,:,1) > 200; QG = Q(:,:,2) > 200; QB = Q(:,:,3) > 200;
        Q = (QR | QB ) .* ~QG ;
        % load the 3-class annotation
        C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        bootstrap = zeros(size(STATS)); clear labels;
        for l=1:length(STATS)
            bootstrap(l) = mode(Q(STATS(l).PixelIdxList) );
            labels(l) = mode(C(STATS(l).PixelIdxList) );
        end
        
        % construct the featureVector we train with!
        featureVector = [H];
        
        cls2 = find(labels == 2);
        cls1 = find(labels == 1);
        negB = find(labels == 0);
        negBH = find(   (labels' == 0) & (RAYFEATUREVECTOR(:,1) > 150) );
        negBL = find(   (labels' == 0) & (RAYFEATUREVECTOR(:,1) <= 150) );
        negQ = find(bootstrap == 1);  NBH = round(.4*N0); NBL = round(.4*N0);  NQ = N0 - NBH - NBL;
        
        clear RAYFEATUREVECTOR H;
        
        % sample the lists
        c2list = randsample(cls2, N2)';
        c1list = randsample(cls1, N1)';
        nlistBH = randsample(negBH, NBH)';
        nlistBL = randsample(negBL, NBL)';
        nlistQ = randsample(negQ, NQ)';
        
        TRAIN = [TRAIN; ...
                 featureVector(c2list,:); ...
                 featureVector(c1list,:);
                 featureVector(nlistBH,:); ...
                 featureVector(nlistBL,:); ...
                 featureVector(nlistQ,:)]; %#ok<AGROW>
        TRAIN_L = [TRAIN_L; ...
                   MITO_LABEL*ones(N2,1); ...
                   BND_LABEL*ones(N1,1); ...
                   zeros(N0,1)]; %#ok<AGROW>
    end
    
    
    
    % rescale the data
    T1 = TRAIN; limits = zeros(size(D));
    for x = 1:size(D,1)
        limits(x,:) = [min(min(TRAIN(:,D(x,1):D(x,2)))) max(max(TRAIN(:,D(x,1):D(x,2))))];
        TRAIN(:,D(x,1):D(x,2)) = mat2gray(TRAIN(:,D(x,1):D(x,2)), limits(x,:));
    end
    
     %       keyboard;
    
    
    %% =========== select parameters for the SVM =========================
    disp('Selecting parameters for the SVM');
    bestcv = 0;  
    CMIN = -1; CMAX = 3;  GMIN = -4; GMAX = 1;
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

    
    
    %% ============== loop through the test images and do prediction ======
    for i = testImgs
        disp(['Predicting for ' d(i).name]);
        fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
        fileRoot = fileRoot{1};
        % load the RAY features and the labels
        load([featureFolder d(i).name]); 
        % load the Hist features
        [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
        H = full(H);
        % construct the featureVector we train with!
        featureVector = [H];
        clear RAYFEATUREVECTOR H;
        % load the 3-class annotation
        C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        bootstrap = zeros(size(STATS)); clear labels;
        for l=1:length(STATS)
            bootstrap(l) = mode(Q(STATS(l).PixelIdxList) );
            labels(l) = mode(C(STATS(l).PixelIdxList) );
        end
        labels = labels(:); clear mito;
        
        I = imread([imgFolder fileRoot '.png']);
        
        % normalize the data 
        for x = 1:size(D,1)
            featureVector(:,D(x,1):D(x,2)) = mat2gray(featureVector(:,D(x,1):D(x,2)), limits(x,:));
        end
    
        % perform the SVM prediction
        cmd = '-b 1';
        [pre_L, acc, probs] = svmpredict(labels, featureVector, model, cmd);
    
        % display the image
        disp('writing the prediction image');
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        mito_label_index = find(model.Label == 2); P2 = zeros(size(I)); P1 = P2; P0=P1; Pv=P1; bnd_label_index = find(model.Label == 1); bg_label_index = find(model.Label == 0);
        for s = superpixels
            P2(STATS(s).PixelIdxList) = probs(s,mito_label_index);
            P1(STATS(s).PixelIdxList) = probs(s,bnd_label_index);
            P0(STATS(s).PixelIdxList) = probs(s,bg_label_index);
            [y maxind] = max(probs(s,:));
            Pv(STATS(s).PixelIdxList) = maxind;
        end
        resultIM2 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P2)));
        imwrite(resultIM2,  [destinationFolder fileRoot '_c2.png'], 'PNG');
        resultIM1 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P1)));
        imwrite(resultIM1,  [destinationFolder fileRoot '_c1.png'], 'PNG');
        resultIM0 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P0)));
        imwrite(resultIM0,  [destinationFolder fileRoot '_c0.png'], 'PNG');
        
        % write the predictions to a text file
        disp('writing the prediction text file');
        writePrediction3class(destinationFolder, [fileRoot '.txt'], probs, pre_L, model.Label);
        
        % check segmentation result with annotation
        %Pv = (P2(:) > .5) || (P1(:) > .5);
        Pv = Pv ~=0;
        A = imread([annotationFolder fileRoot '.png']); A = A(:,:,2) > 200;
        ACC = rocstats(Pv, A(:), 'ACC');
        disp([num2str(ACC*100) '% accuracy on ' fileRoot]);
        fid = fopen([destinationFolder 'results.txt'], 'a'); fprintf(fid, '%g accuracy on %s\n', ACC*100, fileRoot); fclose(fid);
    end
end