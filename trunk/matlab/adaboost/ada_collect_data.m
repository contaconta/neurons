function SET = ada_collect_data(DATASETS, set_type, varargin)
%ADA_COLLECT_DATA organizes training images for viola-jones
%
%   SET = ada_collect_data(path1, path0, ...) collects and processes 
%   training images found in path1 (positive training class) and path0
%   (negative training class).  The data is stored into struct SET for
%   use in other vj modules.  Optional argument 'SAVE' will save SET to
%   a specified filename.  Optional argument 'IMRESIZE' allows the user to
%   define the training image size [WIDTH HEIGHT] (all images will be 
%   resized to a default size of [24 24]).  Optional argument 'normalize' 
%   toggles if the image histogram should be normalized (1=yes/0=no).
%   Optional argument 'DATA_LIMIT' will select a subset of the first
%   [POS_EXAMPLES NEG_EXAMPLES] from the training set.  Optional argument
%   'v' for verbose output.
%
%   TRAIN       = ada_collect_data(DATASETS, 'train');
%   VALIDATION  = ada_collect_data(DATASETS, 'validation')
%   TRAIN       = ada_collect_data(DATASETS, 'update', TRAIN, CASCADE, LEARNERS);
%   VALIDATION  = ada_collect_data(DATASETS, 'update', VALIDATION, CASCADE, LEARNERS);
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, ADA_TRAIN, INTEGRAL_IMAGE, ADA_ADABOOST
 


% collect settings from DATASETS 
%   NORM = normalize images?, 
%   IMSIZE = training image size, 
%   POS_LIM = # positive examples, 
%   NEG_LIM = # of negative examples.

[NORM IMSIZE POS_LIM NEG_LIM] = collect_arguments(DATASETS, set_type);
count = 1;



%% initial collection: POSITIVE (c = 1) and NEGATIVE (c = 2) images into SET
if  ~strcmp(set_type, 'update')   %nargin == 2
    for c = 1:2  % the 2 classes
        % collect the training image files into d, and initialize the data struct
        if c == 1
            d = ada_trainingfiles(DATASETS.filelist, set_type, '+', POS_LIM);
        else
            d = ada_trainingfiles(DATASETS.filelist, set_type, '-', NEG_LIM);
        end

        % add each image file to SET, format it, normalize it, and compute features
        for i = 1:length(d)
            % read the file
            filenm = d{i};
            I = imread(filenm);

            % convert to proper class (pixel intensity represented by [0,1])
            if ~isa(I, 'double')
                cls = class(I);
                I = mat2gray(I, [0 double(intmax(cls))]); 
            end

            % convert to grasyscale if necessary
            if size(I,3) > 1
                I = rgb2gray(I);
            end

            % resize to standard size
            if ~isequal(size(I), IMSIZE)
                I = imresize(I, IMSIZE);
            end

            % normalize if necessary
            if NORM
                I = imnormalize('image', I);
            end

            SET.Images(:,:,count) = I;
            if c == 1; SET.class(count) = 1; end
            if c == 2; SET.class(count) = 0; end
            
            count = count + 1;       
        end
    end
end



%% update collection: TRUE NEGATIVE examples are updated with False Positives.
% also:  PRECOMPUTE the feature responses to these new examples.
if strcmp(set_type, 'update')

    SET = varargin{1};
    DETECTOR = varargin{2};
    LEARNERS = varargin{3};
    DELTA = DATASETS.delta;     % how many pixels to skip by default when scanning
    
    % construct a list of files to search in
    d = ada_trainingfiles(DATASETS.filelist, 'update', '-');   
    
    % find a list of the current set of True Negatives in the data set
    TN_LIST = get_true_negatives(SET, DETECTOR);
    FPs_REQUIRED = length(TN_LIST);

    % Collect FPs-REQUIRED images containing false positives
    [FP_LIST, success] = randomscan(d, IMSIZE, NORM, DETECTOR, LEARNERS, DATASETS, FPs_REQUIRED);
    if ~success
        disp('       ...randomly scanning was progressing too slow, deterministically scanning through all images to find FP examples.');
        FP_LIST = rasterscan(d, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST, FPs_REQUIRED);
    end
    
   
    % <===== HERE IS WHERE I NEED TO WORK FROM!!!
        
    
    % Replace the True Negatives with newly collected False Positives
    for i = 1:length(TN_LIST)
        SET.Images(:,:,TN_LIST(i)) = FP_LIST{i};
%         disp(['replaced TN ' num2str(TN_LIST(i)) ' with an FP.']);
%         imshow(FP_LIST{i});
%         pause(0.1);
    end
        
end  
    
        

%% alphabetize the fields of the set
SET = orderfields(SET);   






%% =========================== SUB - FUNCTIONS ======================================
%
%   randomscan -> searches images randomly to find FPs
%   rasterscan -> scans each image in raster fashon to find FPs
%   get_true_negatives ->  returns a list of images in the set which
%                          produce TN detector responses
%   collect_arguments ->   collects settings from DATASETS 
%   update_bigmatrix -> precomputes feature values for newly found FPs
%   


function [FP_LIST, success] = randomscan(d, IMSIZE, NORM, DETECTOR, LEARNERS, DATASETS, FPs_REQUIRED)

success = 0;  find_rate = 1; attempts = 1; FP_LIST = {};
FIND_RATE_LIMIT = .0001;                        %  minimum rate to find FP examples
disp('       ...randomly scanning for FP examples');

while length(FP_LIST) < FPs_REQUIRED
    % 1. randomly select a file from the list
    file_ind = randsample(1:length(d),1);
    filenm = d{file_ind}; I = imread(filenm);  %disp(['scanning ' filenm]);
    
    % convert to grasyscale if necessary
    if size(I,3) > 1
        I = rgb2gray(I);
    end
    
    % 2. randomly select a location and window size from the image
    if isfield(DATASETS, 'scale_limits')
        W = randsample(round(IMSIZE(2) * scale_selection(I,IMSIZE, 'limits', DATASETS.scale_limits)),1);  
        H = round( (IMSIZE(1)/IMSIZE(2)) * W);
    else
        W = randsample(round(IMSIZE(2) * scale_selection(I,IMSIZE)),1);  
        H = round( (IMSIZE(1)/IMSIZE(2)) * W);
    end
    X = ceil(  (size(I,2) - W)  * rand(1));
    Y = ceil(  (size(I,1) - H)  * rand(1));
    
    % 3. sample from this image, location, size
    rect = [X Y W-1 H-1];
    Image = imcrop(I, rect);
    
    % 4. adjust the image as necessary
    if ~isa(Image, 'double')
        cls = class(Image); Image = mat2gray(Image, [0 double(intmax(cls))]); 
    end
    Image = imresize(Image, IMSIZE);
    
    % normalize if necessary
    if NORM
        Image = imnormalize('image', Image);
    end

    % classify the example
    C = ada_classify_individual(DETECTOR, Image, LEARNERS);

    if C
        %disp(['added (random scan) false positive ' num2str(length(FP_LIST)+1) '  current FP rate = ' num2str(find_rate)]);
        FP_LIST{length(FP_LIST)+1} = Image;
    end
    
    % 6. check to see if find_rate is too, if raster scan is required   
    find_rate = length(FP_LIST) / attempts;  %     disp(['current FP finding rate = ' num2str(find_rate)]);
    
    if (attempts > 1000) && (find_rate < FIND_RATE_LIMIT)
        return
    end

    attempts = attempts + 1;
end

% if we get to this point, we were successful in finding false positives!
success = 1;
disp(['       ...found FP examples at a rate of = ' num2str(find_rate*100) '%']);







function FP_LIST = rasterscan(d, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST, FPs_REQUIRED)

%while (length(FP_LIST) < FPs_REQUIRED) && (end_not_reached)
    for file_ind = 1:length(d)

        % read the file
        filenm = d{file_ind}; I = imread(filenm);  disp(['scanning ' filenm]);

        % convert to proper class (pixel intensity represented by [0,1])
        if ~isa(I, 'double')
            cls = class(I); I = mat2gray(I, [0 double(intmax(cls))]); 
        end

        % convert to grasyscale if necessary
        if size(I,3) > 1
            I = rgb2gray(I);
        end

        if isfield(DATASETS, 'scale_limits')
            scales = scale_selection(I, IMSIZE, 'limits', DATASETS.scale_limits);
        else
            scales = scale_selection(I, IMSIZE);
        end

        % loop through the scales
        for scale = scales
            Iscaled = imresize(I, scale);
            actual_scale = size(Iscaled,1) / size(I,1);
            disp(['detector scale = ' num2str(1/actual_scale)]);
            W = size(Iscaled,2);  H = size(Iscaled,1);

            DS = round(DELTA*actual_scale);
            disp(['scaled delta = ' num2str(DS)]);

            % scan the image at each scale
            for c = 1:max(1,DS):W - IMSIZE(2)
                for r = 1:max(1,DS):H - IMSIZE(1)

                    Image = Iscaled(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1);

%                     figure(12343); Itemp = Iscaled; Itemp(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1) = ones(size(Iscaled(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1)));
%                     imshow(Itemp);  pause(.01); refresh; 
%                     disp(['scanning (' num2str(r) ',' num2str(c) ')']);

                    % normalize if necessary
                    if NORM
                        Image = imnormalize('image', Image);
                    end

                    % classify the example
                    C = ada_classify_individual(DETECTOR, Image, LEARNERS);

                    if C
                        %disp(['added (raster scan) false positive ' num2str(length(FP_LIST)+1) ]);
                        FP_LIST{length(FP_LIST)+1} = Image;
                    end
                    
                    % if we've collected enough FPs, return.
                    if length(FP_LIST) >= FPs_REQUIRED
                        return;
                    end
                    
                end
            end
        end
    end
%     end_not_reached = 0;
% end


function TN_LIST = get_true_negatives(SET, DETECTOR)

C = ada_classify_set(DETECTOR, SET);

TN_LIST = (C == SET.class) .* ~SET.class;

TN_LIST = find(TN_LIST);

disp(['       ...found ' num2str(length(TN_LIST)) ' TNs and ' num2str( length(find(SET.class == 0)) - length(TN_LIST)) ' FPs in previous data set ']);


function [NORM IMSIZE POS_LIM NEG_LIM] = collect_arguments(DATASETS, set_type)

%% define default parameters
NORM = 1; IMSIZE = [24 24];
POS_LIM = Inf;NEG_LIM = Inf;

%% collect predefined arguments
if isfield(DATASETS, 'IMSIZE')
    if ~isempty(DATASETS.IMSIZE)
        IMSIZE = DATASETS.IMSIZE;
    end
end

if isfield(DATASETS, 'NORMALIZE')
    if ~isempty(DATASETS.NORMALIZE)
        NORM = DATASETS.NORMALIZE;
    end
end

if strcmp(set_type,'train')
    if isfield(DATASETS, 'TRAIN_POS')
        if ~isempty(DATASETS.TRAIN_POS)
            POS_LIM = DATASETS.TRAIN_POS;
        end
    end
    if isfield(DATASETS, 'TRAIN_NEG')
        if ~isempty(DATASETS.TRAIN_NEG)
            NEG_LIM = DATASETS.TRAIN_NEG;
        end
    end
end
    
if strcmp(set_type,'validation')
    if isfield(DATASETS, 'VALIDATION_POS')
        if ~isempty(DATASETS.VALIDATION_POS)
            POS_LIM = DATASETS.VALIDATION_POS;
        end
    end
    if isfield(DATASETS, 'VALIDATION_NEG')
        if ~isempty(DATASETS.VALIDATION_NEG)
            NEG_LIM = DATASETS.VALIDATION_NEG;
        end
    end
end


% function update_bigmatrix(TN_LIST, SET, LEARNERS, WEAK)
% 
% 
% for l = 1:length(LEARNERS)
%     switch LEARNERS(l).feature_type
%     
%         case 'haar'
%             %% haar like weak learners - for haars it is faster to compute the
%             %  haar response of a single feature over all training examples
% 
%             % collect all the vectorized integral images into IIs
%             for i = 1:length(TN_LIST)
%                 II = integral_image(SET.Images(:,:,TN_LIST));
%                 IIs(:,i) = II(:);
%             end
%             
%             % compute the number of haar features
%             num_haars = 0;
%             for f = 1:length(WEAK.learners)
%                 if strcmp(WEAK.learners{f}.type, 'haar')
%                     num_haars = num_haars + 1;
%                 end
%             end
% 
%             W = wristwatch('start', 'end', num_haars, 'every', 10000, 'text', '    ...precomputed haar feature ');
%             block = round(FILES.memory / (length(TN_LIST)*SET.responses.bytes));        % block = # columns fit into memory lim
%             R = zeros(length(TN_LIST), block );                       % R temporary feature response storage
%             j = 1;                                                      % feature index in current block
%             f_list = [];
%             
%             % loop through features, compute response of feature f to all
%             % examples, store as columns
%             
%             for f = 1:length(WEAK.learners)
%                 if strcmp(WEAK.learners{f}.type, 'haar')
%                     R(:,j) = ada_haar_response(WEAK.learners{f}.hinds, WEAK.learners{f}.hvals, IIs);
%                     f_list = [f_list f];
% 
%                     if mod(f,block) == 0
%                         disp(['    ...writing to ' SET.responses.filename]);
%                         rows = 1:length(SET.class);
%                         cols = f_list;
%                         SET.responses.storeBlock(R,rows,cols);
%                         j = 0;
%                         f_list = [];
%                     end
%                     W = wristwatch(W, 'update', f);
%                     j = j + 1;
%                 end
%             end
%             
%             % store the last columns
%             disp(['    ...writing to ' SET.responses.filename]);
%             %cols = WEAK.learners{1}{3}(f-j+2:f);
%             cols = f_list;
%             %A = R(:,1:j-1);
%             SET.responses.storeBlock(R(:,1:j-1),rows,cols);
%             
%             clear R;
% 
%         %% spedge weak learners
%         %  unlike haars, spedges are faster to compute by looping through
%         %  the examples and computing all spedges for each example.
%         case 'spedge'
%             
%             % create a list of spedge feature indexes
%             f_list = []; 
%             for f = 1:length(WEAK.learners)
%                 if strcmp(WEAK.learners{f}.type, 'spedge')
%                     f_list = [f_list f];
%                 end
%             end
%             
%             
%             %block = round(FILES.memory / (length(WEAK.learners{l}{3})*4)); 
%             block = min(length(SET.class), round(FILES.memory / (length(f_list)*SET.responses.bytes))); 
%             W = wristwatch('start', 'end', length(SET.class), 'every', 200, 'text', '    ...precomputed spedge for example ');
%             %R = zeros(block, length(WEAK.learners{l}{3}));
%             R = zeros(block, length(f_list));
%             j = 1;
%             
%             
% 
%             % loop through examples, compute all spedge repsonses for
%             % example i, store as rows
%             for i = 1:length(SET.class)
%             
%                 sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).sigma);
%                 R(j,:) = sp.spedges(:);
%                 
%                 if mod(i,block) == 0
%                     disp(['    ...writing to ' SET.responses.filename]);
%                     rows = i-block+1:i;
%                     %cols = WEAK.learners{l}{3}(:);
%                     cols = f_list;
%                     SET.responses.storeBlock(R, rows, cols);
%                     j = 0;
%                 end
%                 W = wristwatch(W, 'update', i);
%                 j = j + 1;
%             end
%             
%             if j ~= 1
%                 % store the last rows, if we have some left over
%                 disp(['    ...writing to ' SET.responses.filename]);
%                 rows = i-j+2:i;
%                 SET.responses.storeBlock(R(1:j-1,:), rows, cols);
%                 clear R;
%             end
%     end
% end



% function FP_LIST = get_old_FPs(SET, DETECTOR)
% 
% % disp('   ...collecting FPs from previous data set');
% % 
% % FP_LIST = [];
% % N = SET([SET.class] == 0);
% % 
% % for i = 1:length(N)
% % 
% %     %C = ada_classify_cascade(DETECTOR,  N(i), [0 0]);
% %     C = ada_classify_set(DETECTOR, SET);
% % 
% %     if C
% %         FP_LIST = [FP_LIST N(i)];
% %     end
% %     
% % end
% % 
% % disp(['   ...found ' num2str(length(FP_LIST)) ' FPs in previous data set ']);
% 
% 
% 
% C = ada_classify_set(DETECTOR, SET);
% 
% FP_LIST = (C ~= SET.class) .* ~SET.class;
% 
% FP_LIST = find(FP_LIST);
% 
% disp(['   ...found ' num2str(length(FP_LIST)) ' FPs in previous data set ']);
% 









% 
% function S = scan_next(d, IMSIZE, DELTA, NORM, DATASETS, method, Slast)
% 
% 
% % select the file index we need to load
% switch method
%     case 'random'
%         S.file_ind = randsample(1:length(d),1);       
%     case 'raster'
%         S.file_ind = Slast.file_ind;   
% end
% 
% % read the selected file
% filenm = d{file_ind}; I = imread(filenm);  disp(['scanning ' filenm]);
% 
% % convert to grasyscale if necessary
% if size(I,3) > 1; I = rgb2gray(I); end
% 
% % determine the correct list of scales this image must be scanned at
% scales = scale_selection(I, IMSIZE, 'limits', DATASETS.scale_limits);
% 
% switch method
%     case 'random'
%         S.scale = 
%     case 'raster'
%         S.scale = 
% 


