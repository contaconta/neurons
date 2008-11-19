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
%   TRAIN = ada_collect_data(DATASETS, LEARNERS, 'train');
%   TRAIN = ada_collect_data(DATASETS, LEARNERS, 'train', 'update', TRAIN, CASCADE);
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, ADA_TRAIN, INTEGRAL_IMAGE, ADA_ADABOOST
 
% collect settings from DATASETS
[NORM IMSIZE POS_LIM NEG_LIM] = collect_arguments(DATASETS, set_type);
count = 1;

%% initial collection: POSITIVE (c = 1) and NEGATIVE (c = 2) images into SET
if nargin == 2
    for c = 1:2  % the 2 classes
        % collect the training image files into d, and initialize the data struct
        if c == 1
            d = ada_trainingfiles(DATASETS.filelist, set_type, '+', POS_LIM);
            %SET.Images(length(d)) = [];
        else
            d = ada_trainingfiles(DATASETS.filelist, set_type, '-', NEG_LIM);
            %SET.Images(length(SET) + length(d)) = [];
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

           
                    
%                 if strcmp(WEAK.learners{1}{1}, 'haar')
%                     II = integral_image(I);   
%                     
%                     
%                     
%                 end
%                 
%                 if strcmp(WEAK.learners{1}{1}, 'spedge')
%                     sp = spedges(I, LEARNERS(l).angles, LEARNERS(l).sigma);
%                     spbar = sp.spedges(:);
%                 end
%             end
                
%             for j = 1:length(WEAK.list)
% 
%                 if strcmp(WEAK.list{j,1}, 'haar')
%                     field = WEAK.list{j,1};
%                     ind = WEAK.list{j,2};
%                     RESPONSES(j) = ada_haar_response(WEAK.(field)(ind).hinds, WEAK.(field)(ind).hvals, II);
%                 end
% 
%                 if strcmp(WEAK.list{j,1}, 'spedge')
%                     %field = WEAK.list{j,1};
%                     ind = WEAK.list{j,2};
%                     RESPONSES(j) = spbar(ind);
%                 end
%             end
                
                    
%                     haar_list = [];
%                     for i = 1:length(WEAK.learners)
%                         if strcmp('haar', WEAK.learners{i}{1})
%                             haar_list = [haar_list WEAK.learners{i}{3}];
%                         end
%                     end
%                     
%                     for i = haar_list
%                         field = WEAK.list{i,1};
%                         ind = WEAK.list{i,2};
%                         f_responses(j,:) = ada_haar_response(WEAK.(field)(ind).hinds, WEAK.(field)(ind).hvals, IIs);
%                     
%                     
%                     RESPONSES(WEAK.learners{l}{3}) = II(:);
%                     %SET(count).II = II(:);
%                     %SET.responses.storeCols(II(:),count);
%                 end
%                 if strcmp(LEARNERS(l).feature_type, 'spedge')
%                     sp = spedges(I, LEARNERS(l).angles, LEARNERS(l).sigma);
%                     %SET(count).sp = sp.spedges(:);
%                     %SET.responses.storeCols(sp.spedges(:),count);
%                     RESPONSES(WEAK.learners{l}{3}) = sp.spedges(:);
%                 end
            %end
            
            
            
            count = count + 1;       
        end
    end
end


%% update collection: NEGATIVE examples are updated with False Positives
if nargin > 6

    SET = varargin{2};
    DETECTOR = varargin{3};
    DELTA = DATASETS.delta;     % how many pixels to skip by default when scanning
    
    %% task 1: build a collection of false positive training images
    
    % construct a list of files to search in
    d = ada_trainingfiles(DATASETS.filelist, 'update', '-');   
    
    % randomly sample from the update set to generate a FP_LIST
    FP_LIST = get_old_FPs(SET, DETECTOR);
    [FP_LIST, success] = randomscan(d, IMSIZE, NORM, DETECTOR, LEARNERS, DATASETS, NEG_LIM, FP_LIST);
    if ~success
        disp('    ...randomly scanning was progressing too slow, deterministically scanning through all images to find FP examples.');
        FP_LIST = scanallimages(d, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST);
    end
    
    
    
    %% task 2: randomly select up to NEG_LIM new NEGATIVE examples
    NUM_EXAMPLES = min(NEG_LIM, length(FP_LIST));
    inds = randsample(1:length(FP_LIST),NUM_EXAMPLES);      % sample with replacement from FP_LIST
    
    N_NEW = orderfields(FP_LIST(inds));
    
    %% task 3: replace the current NEGATIVE examples with new FP examples
    P = orderfields(SET( [SET(:).class] == 1));
    SET = [P N_NEW];
    
end  
    
        
SET = orderfields(SET);   


%% SUB - FUNCTIONS



function [FP_LIST, success] = randomscan(d, IMSIZE, NORM, DETECTOR, LEARNERS, DATASETS, NEG_LIM, FP_LIST)

success = 0;  find_rate = 1;
FIND_RATE_LIMIT = .0001;                        %  minimum rate to find FP examples
attempts = 1;
disp('   ...randomly scanning for FP examples');

while length(FP_LIST) < NEG_LIM
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
    EXAMPLE.Image = imcrop(I, rect);
    
    % 4. compute features and adjust the image as necessary
    if ~isa(EXAMPLE.Image, 'double')
        cls = class(EXAMPLE.Image); EXAMPLE.Image = mat2gray(EXAMPLE.Image, [0 double(intmax(cls))]); 
    end

    EXAMPLE.Image = imresize(EXAMPLE.Image, IMSIZE);
    
    % normalize if necessary
    if NORM
        EXAMPLE.Image = imnormalize('image', Example.Image);
    end

    
    
    % compute the appropriate features
    for l = 1:length(LEARNERS)
        if strcmp(LEARNERS(l).feature_type, 'haar')
            II = integral_image(EXAMPLE.Image);
            EXAMPLE.II = II(:);
        end
        if strcmp(LEARNERS(l).feature_type, 'spedge')
            sp = spedges(EXAMPLE.Image, LEARNERS(l).angles, LEARNERS(l).sigma);
            EXAMPLE.sp = sp.spedges(:);
        end
    end
    
    
    % 5. classify
    C = ada_classify_cascade(DETECTOR,  EXAMPLE, [0 0]);

    if C
        EXAMPLE.class = 0;
        FP_LIST = [FP_LIST EXAMPLE];
    end
    
    % 6. check to see if find_rate is too low and we need to do a
    %    deterministic search
    
    find_rate = length(FP_LIST) / attempts;
    %disp(['current FP finding rate = ' num2str(find_rate)]);
    
    if (attempts > 1000) && (find_rate < FIND_RATE_LIMIT)
        return
    end
    
    attempts = attempts + 1;
end

% if we get to this point, we were successful in finding false positives!
success = 1;
disp(['   ...found FP examples at a rate of = ' num2str(find_rate*100) '%']);


function FP_LIST = scanallimages(d, IMSIZE, DELTA, NORM, DETECTOR, LEARNERS, DATASETS, FP_LIST)


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
            EXAMPLE = [];
            for r = 1:max(1,DS):H - IMSIZE(1)

                EXAMPLE.Image = Iscaled(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1);

                %figure(12343); Itemp = Iscaled; Itemp(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1) = ones(size(Iscaled(r:r+IMSIZE(2)-1, c:c + IMSIZE(1) -1)));
                %imshow(Itemp);  pause(.01); refresh; 
                %disp(['scanning (' num2str(r) ',' num2str(c) ')']);
                
                % normalize if necessary
                if NORM
                    EXAMPLE.Image = imnormalize('image', Example.Image);
                end

                % compute the appropriate features
                for l = 1:length(LEARNERS)
                    if strcmp(LEARNERS(l).feature_type, 'haar')
                        II = integral_image(EXAMPLE.Image);
                        EXAMPLE.II = II(:);
                    end
                    if strcmp(LEARNERS(l).feature_type, 'spedge')
                        sp = spedges(EXAMPLE.Image, LEARNERS(l).angles, LEARNERS(l).sigma);
                        EXAMPLE.sp = sp.spedges(:);
                    end
                end

                % classify the EXAMPLE
                C = ada_classify_cascade(DETECTOR,  EXAMPLE, [0 0]);

                if C
                    EXAMPLE.class = 0;
                    FP_LIST = [FP_LIST EXAMPLE];
                end
            end
        end
    end
end







function FP_LIST = get_old_FPs(SET, DETECTOR)

disp('   ...collecting FPs from previous data set');

FP_LIST = [];
N = SET([SET.class] == 0);

for i = 1:length(N)

    C = ada_classify_cascade(DETECTOR,  N(i), [0 0]);

    if C
        FP_LIST = [FP_LIST N(i)];
    end
    
end

disp(['   ...found ' num2str(length(FP_LIST)) ' FPs in previous data set ']);




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

