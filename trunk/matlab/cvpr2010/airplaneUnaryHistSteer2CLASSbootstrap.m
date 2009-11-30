resultname = 'heathrowHistSteer2CLASSboot';

raysName = 'heathrowEdge6';

%boundaryFolder = '/osshare/DropBox/Dropbox/aurelien/superpixels/annotations/';
%histFolder = '/osshare/Work/neurons/matlab/features/rays/featurevectors/airplaneHist/';
histFolder = '/osshare/DropBox/Dropbox/aurelien/FeatureVectors/histogram/heathrow/';
steerableFolder = './featurevectors/heathrowSteerable/';
featureFolder = ['./featurevectors/' raysName '/'];
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrowAnnotations/';
bndFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrowAnnotations/superpixelAnnotations/';
imgFolder = '/osshare/DropBox/Dropbox/aurelien/airplanes/heathrow/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/unary/' resultname '/'];
if ~isdir(destinationFolder); mkdir(destinationFolder); end;

addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');



% k-folds parameters
imgs = 1:13;                % list of image indexes
K = 3;                      % the # examples per fold
TRAIN_LENGTH = 7000;        % the total # of examples per class in training set
AIRPLANE_LABEL = 1;             % label used for mito
%BND_LABEL = 1;              % boundary label



%----------------------------------------------------------------------
D = [];  % Rays30
for x = 1:100;
    D(size(D,1)+1,:) = [x x]; % Intensity
end
%----------------------------------------------------------------------

% %----------------------------------------------------------------------
% D = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
% for x = 105:204
%     D(size(D,1)+1,:) = [x x]; % Intensity
% end
% %----------------------------------------------------------------------


for k =  1:13  % 1:4
    if k == 1; k1 = 1; else; k1 = (k-1)*K +1; end; %#ok<NOSEM>
%     testImgs = imgs( k1:min(k1+K-1, max(imgs)));
%     trainImgs = setdiff(imgs, testImgs);
%     testImgs = 1:13;
%     trainImgs = 1:13;
    testImgs = k;
    trainImgs = setdiff(1:13,k);
    disp(['Testing: ' num2str(testImgs)]);
    disp(['Training: ' num2str(trainImgs)]);
    
    % number of samples per class (N +, N-)
    N = round( TRAIN_LENGTH / length(trainImgs));
   	%N3 = round(.20*N);
    %N2 = round(.30*N);
    N1 = round(.40*N);
    N0 = round(.60*N);
%     N2 = round(.33*N);
%     N1 = round(.33*N);
%     N0 = round(.33*N);
    
    % intialize the training vectors
    TRAIN = [];
    TRAIN_L = [];
    
    % index of the feature data
    d = dir([featureFolder '*.mat']);
    
    % create the training vector looping through training images
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
        % load the normal annotation
        Q = imread([annotationFolder fileRoot '.png']); QR = Q(:,:,1) > 200; QG = Q(:,:,2) > 200; QB = Q(:,:,3) > 200;
        %Q = (QR | QB ) .* ~QG ;
        Q = QG .* ~(QR | QB);
        % load the 3-class annotation
        C = readLabel([bndFolder fileRoot '.label' ], [size(L,1) size(L,2)])'; C = double(C > 0);
        %C = imread([annotationFolder fileRoot '.png' ]); C0 = C(:,:,3) < 200; C1 = C(:,:,1) > 200; C2 = (C(:,:,1) <200 & C(:,:,3) >200); C = zeros(size(C0)) + C1 + 2.*C2;
        
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        bootstrap = zeros(size(STATS)); clear labels;  labels = zeros(size(bootstrap));
        for l=1:length(STATS)
            bootstrap(l) = mode(Q(STATS(l).PixelIdxList) );
            labels(l) = mode(C(STATS(l).PixelIdxList) );
        end
        
        %keyboard;
        
        % construct the featureVector we train with!
        featureVector = [H S];
        
        
        INTENSITY_THRESH = 132;  %median(RAYFEATUREVECTOR(:,1));
        
        %cls2 = find(labels == 2);
        cls1 = find(labels == 1);
        negB = find(labels == 0);
        negBL = find(   (labels == 0) & (RAYFEATUREVECTOR(:,1) <= INTENSITY_THRESH) ); 
        negBH = find(   (labels == 0) & (RAYFEATUREVECTOR(:,1) > INTENSITY_THRESH) ); 
        negBOOT = find(bootstrap == 1);
        
        clear RAYFEATUREVECTOR H S;
        
        %cls3 = find(bootstrap == 1);  
        
        % sample the lists  
        N1 = min(N1, length(cls1));  %N2 = min(N2, length(cls2)); 
        NBL = min(length(negBL), round(.40*N0));  
        NBH = min(length(negBH), round(.40*N0));
        NBOOT = N0 - NBL - NBH;
       
        c1list = randsample(cls1, N1)';
        nlistBH = randsample(negBH, NBH)';
        nlistBL = randsample(negBL, NBL)';
        nlistBOOT = randsample(negBOOT, NBOOT)';

        %keyboard;
        
        TRAIN = [TRAIN; ...
                 featureVector(c1list,:);...
                 featureVector(nlistBOOT,:); ...
                 featureVector(nlistBH,:); ...
                 featureVector(nlistBL,:)]; %#ok<AGROW>
                 %featureVector(c3list,:)]; %#ok<AGROW>
        TRAIN_L = [TRAIN_L; ...
                   AIRPLANE_LABEL*ones(N1,1); ...
                   zeros(N0,1)]; %#ok<AGROW>
                   %BOOTSTRAP_LABEL*ones(N3,1)]; %#ok<AGROW>
    end
   
    
    % rescale the data
    limits = zeros(size(D));
    for x = 1:size(D,1)
        limits(x,:) = [min(min(TRAIN(:,D(x,1):D(x,2)))) max(max(TRAIN(:,D(x,1):D(x,2))))];
        TRAIN(:,D(x,1):D(x,2)) = mat2gray(TRAIN(:,D(x,1):D(x,2)), limits(x,:));
    end
    
    %% =========== select parameters for the SVM =========================
    disp('Selecting parameters for the SVM');
    bestcv = 0;  
    CMIN = -1; CMAX = 3;  GMIN = -4; GMAX = 1;
    %CMIN = -1; CMAX = 3;  GMIN = 0; GMAX = 0;
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
    
%     cmd = ['-b 1 -t 0 -m 500'];
%     model = svmtrain(TRAIN_L, TRAIN, cmd);

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
        %[lab H] = libsvmread([histFolder fileRoot '.fv']);
        [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
        H = full(H);
        [lab S] = libsvmread([steerableFolder fileRoot '_u0_all_feature_vectors']);
        S = full(S);
        
        % construct the featureVector we train with!
        featureVector = [H S ];
        clear RAYFEATUREVECTOR H S;
        
        % load the 3-class annotation
        %C = readLabel([boundaryFolder fileRoot '.label' ], [size(L,1) size(L,2)])';
        %C = imread([annotationFolder fileRoot '.png' ]); C0 = C(:,:,3) < 200; C1 = C(:,:,1) > 200; C2 = (C(:,:,1) <200 & C(:,:,3) >200); C = zeros(size(C0)) + C1 + 2.*C2;
        C = readLabel([bndFolder fileRoot '.label' ], [size(L,1) size(L,2)])'; C = double(C > 0);
        
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        bootstrap = zeros(size(STATS)); clear labels; labels = zeros(size(bootstrap));
        for l=1:length(STATS)
            %bootstrap(l) = mode(Q(STATS(l).PixelIdxList) );
            labels(l) = mode(C(STATS(l).PixelIdxList) );
        end
        labels = labels(:); clear mito;
        
        I = imread([imgFolder fileRoot '.jpg']);  I = rgb2gray(I);
        
        % normalize the data 
        for x = 1:size(D,1)
            featureVector(:,D(x,1):D(x,2)) = mat2gray(featureVector(:,D(x,1):D(x,2)), limits(x,:));
        end
    
        % perform the SVM prediction
        %cmd = '-b 1';
        cmd = '-b 1';
        [pre_L, acc, probs] = svmpredict(labels, featureVector, model, cmd);
    
        % display the image
        disp('writing the prediction image');
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        %mito_label_index = find(model.Label == 2); 
        P2 = zeros(size(I)); P1 = P2; P0=P1; P3=P1; Pv=P1; 
        bnd_label_index = find(model.Label == 1); 
        bg_label_index = find(  (model.Label == 0));
%         boot_label_index = find(model.Label == 3)  ;
        for s = superpixels
%             P3(STATS(s).PixelIdxList) = probs(s,boot_label_index);
            %P2(STATS(s).PixelIdxList) = probs(s,mito_label_index);
            P1(STATS(s).PixelIdxList) = probs(s,bnd_label_index);
            P0(STATS(s).PixelIdxList) = probs(s,bg_label_index);
            [y maxind] = max(probs(s,:));
            Pv(STATS(s).PixelIdxList) = maxind;
        end
%         resultIM2 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P3)));
%         imwrite(resultIM2,  [destinationFolder fileRoot '_c3.png'], 'PNG');
%         resultIM2 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P2)));
%         imwrite(resultIM2,  [destinationFolder fileRoot '_c2.png'], 'PNG');
        resultIM1 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P1)));
        imwrite(resultIM1,  [destinationFolder fileRoot '_c1.png'], 'PNG');
        resultIM0 = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P0)));
        imwrite(resultIM0,  [destinationFolder fileRoot '_c0.png'], 'PNG');
        
        % write the predictions to a text file
        disp('writing the prediction text file');
        writePrediction2class(destinationFolder, [fileRoot '.txt'], probs, pre_L, model.Label);
        
        % check segmentation result with annotation
        %Pv = (P2(:) > .5) || (P1(:) > .5);
        Pv = Pv ~=0;
        A = imread([annotationFolder fileRoot '.png']); A = A(:,:,2) > 200;
        ACC = rocstats(Pv(:), A(:), 'ACC');
        disp([num2str(ACC*100) '% accuracy on ' fileRoot]);
        fid = fopen([destinationFolder 'results.txt'], 'a'); fprintf(fid, '%g accuracy on %s\n', ACC*100, fileRoot); fclose(fid);
    end
    
    
    %keyboard;
end
