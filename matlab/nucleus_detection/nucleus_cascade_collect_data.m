function TRAIN = nucleus_cascade_collect_data(varargin)
%NUCLEUS_COLLECT_DATA organizes training images for viola-jones
%
%   TRAIN = vj_collect_data(path1, path0, ...) 
%   TRAIN = vj_collect_data(path1, path0, TRAIN, CASCADE, ...) 
%   collects and processes training images found in path1 (positive training 
%   class) and path0 (negative training class).  Class 0 training data is 
%   collected from path0 by selecting a random subwindow of a random image.
%   Subsequent calls replace the class 0 data with new randomly selected
%   data which produces a false positive from the CASCADE classifier.
%   Class 1 data in path1 contain previously cropped image. The data is 
%   stored into struct TRAIN.  Optional arguments are summarized below:
%
%   'save', FILENM      saves the TRAIN to a file FILENM
%   'size', [w h]       resizes examples to size [w h] (default [24 24])
%   'normalize', 1      =1 normalizes contrast for each exammple, =0 does not
%   'data_limit', [P N] TRAIN contains P positive and N negative examples
%   'init'              initializes TRAIN (class 1 collected at init only)
%   'v'                 verbose output
%
%   Example 1:
%   collects 100 + example images from path1 and 100 - example images from path0,
%   resized them to [12 12], and does not normalize them.  'initialize' 
%   indicates this is the 1st time we are collecting TRAIN so we do not
%   pass TRAIN, and the positive examples must be collected.
%   --------------
% 
%   TRAIN = vj_cascasde_collect_data(path1, path0, 'size', [12 12], ...
%           'normalize', 0, 'initialize', 'data_limit', [100 100]);
%
%
%   Example 2: now that TRAIN exists, we update new class 0 examples
%   -----------------------------------------------------------------
%   TRAIN = vj_cascasde_collect_data(path1, path0, TRAIN, CASCADE, ...
%                   'size', [12 12], 'normalize', 0, 'data_limit', [100 100]);
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, VJ_TRAIN, INTEGRAL_IMAGE, VJ_ADABOOST
 
% define parameters
path1 = varargin{1};                % positive example path
path0 = varargin{2};                % negative example path
NORM = 1;                           % =1 normalize examples, =0 do not
SAVE = 0;                           % =1 save TRAIN, =0 do not
IMSIZE = [24 24];                   % standard example size
POS_LIM = Inf;                      % default max positive examples
NEG_LIM = Inf;                      % default max negative examples
V = 0;                              % default verbose
INIT = 0;                           % collect positive data when init =1
opt = 1;
W_MU = 18.4271;
W_STD = 5.7483;
W_VAR = sqrt(W_STD);
LOW_VARIANCE_THRESH = .0001; %.005; % minimum variance of a typical negative example
OVERLAP_THRESHOLD = 0.5;            % maximum % of a nucleas that can appear in non-nucleus training image


%% handle optional arguments

if isstruct(varargin{3})
    TRAIN = varargin{3};
    CASCADE = varargin{4};
    opt = 5;
end    
for i = opt:nargin
    if strcmp(varargin{i}, 'size')
        IMSIZE = varargin{i+1};
    end
    if strcmp(varargin{i}, 'normalize')
        NORM = varargin{i+1};
    end
    if strcmp(varargin{i}, 'save')
        SAVE = 1;
        FILENM = varargin{i+1};
    end
    if strcmp(varargin{i}, 'data_limit')
        L = varargin{i+1};
        POS_LIM = L(1);
        NEG_LIM = L(2);
    end
    if strcmp(varargin{i}, 'v')
        V = 1;
    end
    if strcmp(varargin{i}, 'initialize')
        INIT = 1;
    end
    if strcmp(varargin{i}, 'nuclei_locs')
        NUCS = varargin{i+1};
    end
end


%--------------------------------------------------------------------------
%% collect and process the POSITIVE CLASS images only when initializing
%--------------------------------------------------------------------------
if INIT
    d = dir(path1);  d = d(3:length(d));
    count = 1;
    for i = 1:length(d)
        if count <= POS_LIM
            try
                filenm = [path1 d(i).name];
                I = imread(filenm);
                if V; disp(['reading ' filenm ]); end;
            catch  %#ok<CTCH,NASGU>              
            end
            
            if ~isa(I, 'double')
                cls = class(I);
                I = mat2gray(I, [0 double(intmax(cls))]); 
            end
            if size(I,3) > 1
                I = rgb2gray(I);
            end
            if ~isequal(size(I), IMSIZE)
                I = imresize(I, IMSIZE);
            end
            if NORM
                I = imnormalize('image', I);
            end
            II = integral_image(I);
            P(count).Image = I; %#ok<AGROW>
            P(count).II    = II(:); %#ok<AGROW>
            P(count).class = 1; %#ok<AGROW>
            count = count + 1;
        end
    end
else
    P = TRAIN( [TRAIN(:).class] == 1);
end



%--------------------------------------------------------------------------
%% collect new NEGATIVE CLASS images which generate False Positives
%--------------------------------------------------------------------------
% P and N contain the positive and negative examples of the training set.
% we search for examples inside of a dataset of full images until we find
% enough negative examples that produce false positives to continue.

N = [ ];  
for t=1:24; filenm = [path0 'mip' num2str(t) '.png']; ALLI(:,:,t) = imread(filenm); end;
Isize = size(ALLI);

while length(N) < NEG_LIM
    
    % 1. randomly select a large image
    t = ceil(24*rand(1));

    % 2. Randomly sample an EXAMPLE 
    W = round(gsamp(W_MU, W_VAR, 1)); H = W;
    X = ceil(  (Isize(2) - W)  *rand(1));
    Y = ceil(  (Isize(1) - H) * rand(1));
    rect = [X Y W-1 H-1];
    EXAMPLE = imcrop(ALLI(:,:,t), rect);

    % 3. adjust the image if necessary
    if ~isa(EXAMPLE, 'double')
        cls = class(EXAMPLE);
        EXAMPLE = mat2gray(EXAMPLE, [0 double(intmax(cls))]); 
    end
    if size(EXAMPLE,3) > 1
        EXAMPLE = rgb2gray(EXAMPLE);
    end
    
    % 4. get rid of many low variance examples - they occur too often!
    if var(EXAMPLE) < LOW_VARIANCE_THRESH
        if rand(1) < .99
            %disp('too little variance, reselecting.');
            continue;
         end
    end
    
    % 5. Check to make sure we don't overlap an annotation too much
    x1 = rect(1);
    y1 = rect(2);
    x2 = rect(1) + rect(3);
    y2 = rect(2) + rect(4);
    box1 = [x1 y1 x2 y2];
    overlapsGT = 0;
    for i = 1:size(NUCS(t),1)
        box2(1) = NUCS(t).BoundingBoxes(i,1);
        box2(2) = NUCS(t).BoundingBoxes(i,2);
        box2(3) = NUCS(t).BoundingBoxes(i,1) + NUCS(t).BoundingBoxes(i,3);
        box2(4) = NUCS(t).BoundingBoxes(i,2) + NUCS(t).BoundingBoxes(i,4);
        coverageofbox1 = overlap(box1, box2) / (W*H);
        if coverageofbox1 > OVERLAP_THRESHOLD
            overlapsGT = 1;
            break;
        end
    end
    if overlapsGT
        continue;
    end
    
    % 6. normalize if necessary and resize to IMSIZE
    EXAMPLE = imresize(EXAMPLE, IMSIZE);
    if NORM
        EXAMPLE = imnormalize('image', EXAMPLE);
    end

    % 7. Compute the integral image
    II = integral_image(EXAMPLE);


    % 8. use the CASCADE to classify the randomly selected example, and keep it
    %    if it produces a false positive
    if ~INIT
        if vj_classify_cascade(CASCADE, II, [0 0])      % = 1 is a false positive
            N(length(N)+1).Image = EXAMPLE; %#ok<AGROW>
            N(length(N)).II    = II(:); %#ok<AGROW>
            N(length(N)).class = 0; %#ok<AGROW>
        else
            continue
        end
    else    % if we are supposed to initialize N, we don't need to check against the CASCADE
        N(length(N)+1).Image = EXAMPLE; %#ok<AGROW>
        N(length(N)).II    = II(:); %#ok<AGROW>
        N(length(N)).class = 0; %#ok<AGROW>
    end
    if overlapsGT
        disp('I should not print')
    end
end

% merge the positive examples P and negative N into TRAIN
TRAIN = [P N];


% wrap it up by saving
if SAVE
    if V; disp(['saving ' FILENM ' ...']); end;
    save(FILENM, 'TRAINING');
end
