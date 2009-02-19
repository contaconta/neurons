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
TRUE_OVERLAP_THRESH = .2;


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
if strcmp(set_type, 'update')

    SET = varargin{1};
    DETECTOR = varargin{2};
    LEARNERS = varargin{3};
    DELTA = DATASETS.delta;     % how many pixels to skip by default when scanning
    
    % construct a list of files to search in
    d = ada_trainingfiles(DATASETS.filelist, 'update', '-');   
    
    % construct a list of annotations to make sure we only get FP's
    a = ada_trainingfiles(DATASETS.filelist, 'annotation', '+');
    
    % find a list of the current set of True Negatives in the data set
    TN_LIST = get_true_negatives(SET, DETECTOR);
    FPs_REQUIRED = length(TN_LIST);

    % SCAN THE IMAGES!
    disp('       ...raster/random scan.'); FP_LIST =[];
    %FP_LIST = randomscan(d, a, TRUE_OVERLAP_THRESH, IMSIZE, NORM, DETECTOR, LEARNERS, DATASETS, FPs_REQUIRED);
    FP_LIST = rasterscan(d, a, TRUE_OVERLAP_THRESH, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST, FPs_REQUIRED);

    
    % Replace the True Negatives with newly collected False Positives
    for i = 1:length(TN_LIST)
        SET.Images(:,:,TN_LIST(i)) = FP_LIST{i};
%         disp(['replaced TN ' num2str(TN_LIST(i)) ' with an FP.']);        imshow(FP_LIST{i});        pause(0.1);
    end
        
end  
    
% alphabetize the fields of the set
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




function FP_LIST = rasterscan(d, a, TRUE_OVERLAP_THRESH, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST, FPs_REQUIRED)

% randomly permute the file lists, so we don't always start with the same image
rnd = randperm(length(d));    d = d(rnd);  a = a(rnd);  FP_start = length(FP_LIST);  count = 1;


BYTES_LIMIT = 1500000;  % the total # of bytes we will scan in one chunk
w = wristwatch('start', 'end', FPs_REQUIRED, 'every', 100); wstring = '       ...found a new FP #'; tic;

g = 1;  running_bytes = 0; groups = zeros(size(d));
for f = 1:length(d);
    d_file = dir(d{f});
    running_bytes = running_bytes + d_file.bytes;
    groups(f) = g;
    if running_bytes > BYTES_LIMIT;
        g = g + 1;
        running_bytes = 0;
    end
end

for g = 1:max(groups)

    % for each of these files, create a list of the crop rects for a raster
    % scan
 
    group_inds = find(groups == g);
   
    short_filenm_list = d(group_inds);
    short_annotation_list = a(group_inds);
    short_list = [];
    SAFE_BIG_NUMBER = 700000;
    scanlist = zeros(SAFE_BIG_NUMBER,4);  scancount = 1;

    %% construct the scanlist - a short list of files and rects to scan
    for f_ind = 1:length(short_filenm_list)

        I = imread(short_filenm_list{f_ind}); %disp(['       scanning ' short_filenm_list{f_ind}]);
        A = mat2gray(imread(short_annotation_list{f_ind}));

        % convert to proper class (pixel intensity represented by [0,1])
        if ~isa(I, 'double');cls = class(I); I = mat2gray(I, [0 double(intmax(cls))]); end

        % convert to grayscale if necessary
        if size(I,3) > 1; I = rgb2gray(I); end 
        if size(A,3) > 1; A = rgb2gray(A); end

        % store the image into the short_list
        short_list(f_ind).I = I;
        short_list(f_ind).A = A;

        % figure out the scales we will search at
        if isfield(DATASETS, 'scale_limits')
            short_list(f_ind).scales = scale_selection(I, IMSIZE, 'limits', DATASETS.scale_limits);
        else
            short_list(f_ind).scales = scale_selection(I, IMSIZE);
        end
        
        % loop through the scales
        for s = 1:length(short_list(f_ind).scales)
            % resize the image and annotation for scale s
            scale = short_list(f_ind).scales(s);
            short_list(f_ind).Iscaled{s} = imresize(short_list(f_ind).I, scale);
            short_list(f_ind).Ascaled{s} = imresize(short_list(f_ind).A, scale);
            short_list(f_ind).actual_scale(s) = size(short_list(f_ind).Iscaled{s},1) / size(short_list(f_ind).I,1);
        
            % define all of the scan rects at this scale, append to scanlist
            W = size(short_list(f_ind).Iscaled{s},2);  H = size(short_list(f_ind).Iscaled{s},1);
            DS = round(DELTA*short_list(f_ind).actual_scale(s));
        
            if scancount > SAFE_BIG_NUMBER
                disp('too many scan points for SAFE_BIG_NUMBER!');
                keyboard;
            end
            
            for r = 1:max(1,DS):H - IMSIZE(1)
                for c = 1:max(1,DS):W - IMSIZE(2)

                    scanlist(scancount,:) = [f_ind, s, r, c];
                    scancount = scancount + 1;
                end
            end
        end

    end

    %% randomly permute the scanlist
    scanlist = scanlist(1:scancount-1,:); rnd = randperm(scancount-1);
    scanlist = scanlist(rnd,:);
    %disp('randomized the scanlist');
    
    
    %% proceed through the scanlist: crop, classify, and add the example if it produces a false positive
    for i = 1:size(scanlist,1)
        
        f = scanlist(i,1);      % the file index
        s = scanlist(i,2);      % the scale index
        r = scanlist(i,3);      % the row index
        c = scanlist(i,4);      % the col index
        
        Icrop = short_list(f).Iscaled{s}(r:r+IMSIZE(1)-1, c:c + IMSIZE(2)-1);
        Acrop = short_list(f).Ascaled{s}(r:r+IMSIZE(1)-1, c:c + IMSIZE(2)-1);
    
        %figure(124332); imshow(Icrop);  pause(.01); refresh;

        if (sum(Acrop(:)) / numel(Acrop)) > TRUE_OVERLAP_THRESH
            %disp('...oops, we picked a true positive example!');
            count = count + 1; continue;
        end
        
        % normalize if necessary
        if NORM
            Icrop = imnormalize('image', Icrop);
        end
        
        % classify the example
        C = ada_classify_individual(DETECTOR, Icrop, LEARNERS);

        if C
            %disp(['added (raster scan) false positive ' num2str(length(FP_LIST)+1) ]);
            FP_LIST{length(FP_LIST)+1} = Icrop;
            w = wristwatch(w, 'update', length(FP_LIST), 'text', wstring);
        end

        % if we've collected enough FPs, return.
        if length(FP_LIST) >= FPs_REQUIRED
            disp([ num2str(length(FP_LIST) - FP_start) ' new FPs found [' num2str(length(FP_LIST)) '/' num2str(FPs_REQUIRED)  '] at a rate of ' num2str( 100*(length(FP_LIST) - FP_start)/count) '% in ' num2str(toc) ' s']);
            return;
        end

        count = count + 1;
    
    end
        
    %disp('--------------------------------------');
end







% 
% function [FP_LIST, success] = randomscan(d, a, TRUE_OVERLAP_THRESH, IMSIZE, NORM, DETECTOR, LEARNERS, DATASETS, FPs_REQUIRED)
% 
% w = wristwatch('start', 'end', FPs_REQUIRED, 'every', 100); wstring = '       ...found a new FP #';
% success = 0;  find_rate = 1; attempts = 1; FP_LIST = {};
% FIND_RATE_LIMIT = .002;                        %  minimum rate to find FP examples
% disp('       ...randomly scanning for FP examples');
% 
%             
% 
% while length(FP_LIST) < FPs_REQUIRED
%     % 1. randomly select a file from the list
%     file_ind = randsample(1:length(d),1);
%     filenm = d{file_ind}; I = imread(filenm);  %disp(['scanning ' filenm]);
%     filenm = a{file_ind}; A = imread(filenm);
%     
%     % convert to grasyscale if necessary
%     if size(I,3) > 1
%         I = rgb2gray(I);
%     end
%     
%     % 2. randomly select a location and window size from the image
%     if isfield(DATASETS, 'scale_limits')
%         W = randsample(round(IMSIZE(2) * scale_selection(I,IMSIZE, 'limits', DATASETS.scale_limits)),1);  
%         H = round( (IMSIZE(1)/IMSIZE(2)) * W);
%     else
%         W = randsample(round(IMSIZE(2) * scale_selection(I,IMSIZE)),1);  
%         H = round( (IMSIZE(1)/IMSIZE(2)) * W);
%     end
%     X = ceil(  (size(I,2) - W)  * rand(1));
%     Y = ceil(  (size(I,1) - H)  * rand(1));
%     
%     % 3. sample from this image, location, size
%     rect = [X Y W-1 H-1];
%     Image = imcrop(I, rect);
%     
%     % 4. adjust the image as necessary
%     if ~isa(Image, 'double')
%         cls = class(Image); Image = mat2gray(Image, [0 double(intmax(cls))]); 
%     end
%     Image = imresize(Image, IMSIZE);
%     
%     % 5. make sure it does not contain a true positive example
%     Acrop = imcrop(A, rect);
%     if (sum(Acrop(:)) / numel(Acrop)) > TRUE_OVERLAP_THRESH
%         %disp('...oops, we picked a true positive example!');
%         attempts = attempts + 1;
%         continue;
%     end
%     
%     % normalize if necessary
%     if NORM
%         Image = imnormalize('image', Image);
%     end
% 
%     % classify the example
%     C = ada_classify_individual(DETECTOR, Image, LEARNERS);
% 
%     if C
%         %disp(['added (random scan) false positive ' num2str(length(FP_LIST)+1) '  current FP rate = ' num2str(find_rate)]);
%         FP_LIST{length(FP_LIST)+1} = Image;
%         w = wristwatch(w, 'update', length(FP_LIST), 'text', wstring);
%     end
%     
%     % 6. check to see if find_rate is too, if raster scan is required   
%     find_rate = length(FP_LIST) / attempts;  %     disp(['current FP finding rate = ' num2str(find_rate)]);
%     
%     if (attempts > 1000) && (find_rate < FIND_RATE_LIMIT)
%         return
%     end
% 
%     attempts = attempts + 1;
% end
% 
% % if we get to this point, we were successful in finding false positives!
% success = 1;
% disp(['       ...found FP examples at a rate of = ' num2str(find_rate*100) '%']);







% function FP_LIST = rasterscan(d, a, TRUE_OVERLAP_THRESH, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST, FPs_REQUIRED)
% 
% 
% % randomly permute the list, so we don't always start with the same image
% rnd = randperm(length(d));    d = d(rnd); a = a(rnd);
% 
% w = wristwatch('start', 'end', FPs_REQUIRED, 'every', 100); wstring = '       ...found a new FP #';
% tic;
% 
% for file_ind = 1:length(d)
% 
%     % read the file
%     filenm = d{file_ind}; I = imread(filenm);  disp(['       scanning ' filenm]);
%     filenm = a{file_ind}; A = imread(filenm);
% 
%     % convert to proper class (pixel intensity represented by [0,1])
%     if ~isa(I, 'double')
%         cls = class(I); I = mat2gray(I, [0 double(intmax(cls))]); 
%     end
% 
%     % convert to grasyscale if necessary
%     if size(I,3) > 1
%         I = rgb2gray(I);
%     end
% 
%     if isfield(DATASETS, 'scale_limits')
%         scales = scale_selection(I, IMSIZE, 'limits', DATASETS.scale_limits);
%     else
%         scales = scale_selection(I, IMSIZE);
%     end
%     
%     FP_start = length(FP_LIST);  count = 1;
%     tic;
% 
%     % loop through the scales
%     for scale = scales
%         Iscaled = imresize(I, scale);
%         Ascaled = imresize(A, scale);
%         actual_scale = size(Iscaled,1) / size(I,1);
%         %disp(['detector scale = ' num2str(1/actual_scale)]);
%         W = size(Iscaled,2);  H = size(Iscaled,1);
% 
%         DS = round(DELTA*actual_scale);
%         %disp(['scaled delta = ' num2str(DS)]);
% 
%         % scan the image at each scale
%         for c = 1:max(1,DS):W - IMSIZE(2)
%             for r = 1:max(1,DS):H - IMSIZE(1)
% 
%                 Image = Iscaled(r:r+IMSIZE(1)-1, c:c + IMSIZE(2) -1);
% 
% %                     figure(12343); Itemp = Iscaled; Itemp(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1) = ones(size(Iscaled(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1)));
% %                     imshow(Itemp);  pause(.01); refresh; 
% %                     disp(['scanning (' num2str(r) ',' num2str(c) ')']);
% 
%                 
%                 %  make sure it does not contain a true positive example
%                 Acrop = Ascaled(r:r+IMSIZE(1)-1, c:c + IMSIZE(2) -1);
%                 if (sum(Acrop(:)) / numel(Acrop)) > TRUE_OVERLAP_THRESH
%                     %disp('...oops, we picked a true positive example!');
%                     count = count + 1;
%                     continue;
%                 end
% 
%                 % normalize if necessary
%                 if NORM
%                     Image = imnormalize('image', Image);
%                 end
% 
%                 % classify the example
%                 C = ada_classify_individual(DETECTOR, Image, LEARNERS);
% 
%                 if C
%                     %disp(['added (raster scan) false positive ' num2str(length(FP_LIST)+1) ]);
%                     FP_LIST{length(FP_LIST)+1} = Image;
%                     w = wristwatch(w, 'update', length(FP_LIST), 'text', wstring);
%                 end
% 
%                 % if we've collected enough FPs, return.
%                 if length(FP_LIST) >= FPs_REQUIRED
%                     disp([ num2str(length(FP_LIST) - FP_start) ' new FPs found [' num2str(length(FP_LIST)) '/' num2str(FPs_REQUIRED)  '] at a rate of ' num2str((length(FP_LIST) - FP_start)/count) '% in ' num2str(toc) ' s']);
%                     %disp([ num2str(length(FP_LIST)) ' Total FPs found.']);
%                     return;
%                 end
% 
%                 count = count + 1;
%             end
%         end
%     end
%     disp([ num2str(length(FP_LIST) - FP_start) ' new FPs found [' num2str(length(FP_LIST)) '/' num2str(FPs_REQUIRED)  '] at a rate of ' num2str((length(FP_LIST) - FP_start)/count) '% in ' num2str(toc) ' s']);
%     %disp([ num2str(length(FP_LIST)) ' Total FPs found.']);
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
