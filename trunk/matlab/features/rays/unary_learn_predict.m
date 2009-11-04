
resultname = 'Rays30MedInvE2+Hist';

raysName = 'rays30MedianInvariantE2';


histFolder = '/osshare/DropBox/Dropbox/aurelien/histogramFeatureVectors/';
featureFolder = ['./featurevectors/' raysName '/'];
annotationFolder = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';
imgFolder = '/osshare/Work/Data/LabelMe/Images/fibsem/';
destinationFolder = ['/osshare/DropBox/Dropbox/aurelien/shapeFeatureVectors/' resultname '/'];
if ~isdir(destinationFolder); mkdir(destinationFolder); end;

addpath('/home/smith/bin/libsvm-2.89/libsvm-mat-2.89-3/');



% k-folds parameters
imgs = 1:23;                % list of image indexes
K = 5;                      % the # of folds in k-fold training
TRAIN_LENGTH = 4000;        % the total # of examples per class in training set




for k = 1:5
    % determine our training and testing images for this k-fold
    if k == 1; k1 = 1; else; k1 = (k-1)*K +1; end; %#ok<NOSEM>
    testImgs = imgs( k1:min(k1+5-1, max(imgs)));
    trainImgs = setdiff(imgs, testImgs);
    disp(['Testing: ' num2str(testImgs)]);
    disp(['Training: ' num2str(trainImgs)]);
    
    % number of samples per class (N +, N-)
    N = round( TRAIN_LENGTH / length(trainImgs));
    
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
        
        % construct the featureVector we train with!
        featureVector = full([RAYFEATUREVECTOR H]);
        clear RAYFEATUREVECTOR H;
        
        pos = find(labels == 1);
        neg = find(labels == 0);
        plist = randsample(pos, N)';
        nlist = randsample(neg, N)';
        
        TRAIN = [TRAIN; featureVector(plist,:) ; featureVector(nlist,:)]; %#ok<AGROW>
        TRAIN_L = [TRAIN_L; 2*ones(size(plist)); zeros(size(nlist))]; %#ok<AGROW>
    end
    
    %----------------------------------------------------------------------
    DEPEND = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
    for x = 105:124
        DEPEND(size(DEPEND,1)+1,:) = [x x]; % Intensity
    end
    %----------------------------------------------------------------------
    
    % rescale the data
    T1 = TRAIN; limits = zeros(size(DEPEND));
    for x = 1:size(DEPEND,1)
        limits(x,:) = [min(min(TRAIN(:,DEPEND(x,1):DEPEND(x,2)))) max(max(TRAIN(:,DEPEND(x,1):DEPEND(x,2))))];
        TRAIN(:,DEPEND(x,1):DEPEND(x,2)) = mat2gray(TRAIN(:,DEPEND(x,1):DEPEND(x,2)), limits(x,:));
    end
    
    
    %% select parameters for the SVM
    disp('Selecting parameters for the SVM');
    bestcv = 0;
    for log2c = -1:3,
      for log2g = -4:1,
        cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g) ' -m 500'];
        cv = svmtrain(TRAIN_L, TRAIN, cmd);
        if (cv >= bestcv),
          bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
      end
    end
    
    
    %% train the SVM
    disp('Training the best SVM');
    cmd = ['-b 1 -c ' num2str(bestc) ' -g ' num2str(bestg) ' -m 500'];
    model = svmtrain(TRAIN_L, TRAIN, cmd);
    
    %% save the SVM model
    save([destinationFolder 'svm_model' num2str(testImgs) '.mat'], 'model', 'limits', 'DEPEND');
    
    
    
    %% loop through the test images and do prediction
    for i = testImgs
        disp(['Predicting for ' d(i).name]);
        fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
        fileRoot = fileRoot{1};
        % load the RAY features and the labels
        load([featureFolder d(i).name]); 
        % load the Hist features
        [lab H] = libsvmread([histFolder fileRoot '_u0_all_feature_vectors']);
        
        % construct the featureVector we train with!
        featureVector = full([RAYFEATUREVECTOR H]);
        clear RAYFEATUREVECTOR H;
        
        % get the labels
        if length(mito) ~= size(featureVector,1)
            mito(length(mito):size(featureVector,1),1) = 0;  %<<<<<< TEMPORARY FIX FOR BAD MITO
        end
        labels = mito; labels = labels(:); clear mito;
        
        I = imread([imgFolder fileRoot '.png']);
        
        %------------------------------------------------------------------
        DEPEND = [1 1; 2 2; 3 14; 15 26; 27 38; 39 104;];  % Rays30
        for x = 105:124
            DEPEND(size(DEPEND,1)+1,:) = [x x]; % Intensity
        end
        %------------------------------------------------------------------

        % normalize the data 
        for x = 1:size(DEPEND,1)
            featureVector(:,DEPEND(x,1):DEPEND(x,2)) = mat2gray(featureVector(:,DEPEND(x,1):DEPEND(x,2)), limits(x,:));
        end
    
        % perform the SVM prediction
        cmd = '-b 1';
        [pre_L, acc, probs] = svmpredict(labels, featureVector, model, cmd);
    
        % display the image
        disp('writing the prediction image');
        STATS = regionprops(L, 'PixelIdxlist', 'Centroid', 'Area');
        mito_label_index = find(model.Label == 2); P = zeros(size(I));
        for s = superpixels
            P(STATS(s).PixelIdxList) = probs(s,mito_label_index);
        end
        resultIM = imlincomb(.70, mat2gray(gray2rgb(I)), .30, mat2gray(mat2rgb(P)));
        imwrite(resultIM,  [destinationFolder fileRoot '.png'], 'PNG');
        
        % write the predictions to a text file
        disp('writing the prediction text file');
        writePrediction(destinationFolder, [fileRoot '.txt'], probs, pre_L, model.Label);
        
        % check segmentation result with annotation
        P1 = P(:) > .5;
        A = imread([annotationFolder fileRoot '.png']); A = A(:,:,2) > 200;
        ACC = rocstats(P1, A(:), 'ACC');
        disp([num2str(ACC*100) '% accuracy on ' fileRoot]);
        fid = fopen([destinationFolder 'results.txt'], 'a'); fprintf(fid, '%g accuracy on %s\n', ACC*100, fileRoot); fclose(fid);
    end
end