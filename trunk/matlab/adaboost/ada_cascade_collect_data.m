function TRAIN = ada_cascade_collect_data(varargin)
%ADA_COLLECT_DATA organizes training images for viola-jones
%
%   TRAIN = ada_collect_data(path1, path0, ...) 
%   TRAIN = ada_collect_data(path1, path0, TRAIN, CASCADE, ...) 
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
%   TRAIN = ada_cascasde_collect_data(path1, path0, 'size', [12 12], ...
%           'normalize', 0, 'initialize', 'data_limit', [100 100]);
%
%
%   Example 2: now that TRAIN exists, we update new class 0 examples
%   -----------------------------------------------------------------
%   TRAIN = ada_cascasde_collect_data(path1, path0, TRAIN, CASCADE, ...
%                   'size', [12 12], 'normalize', 0, 'data_limit', [100 100]);
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, INTEGRAL_IMAGE, ADA_ADABOOST
 
%% define parameters
path1 = varargin{1};                % positive example path
path0 = varargin{2};                % negative example path
NORM = 1;                           % =1 normalize examples, =0 do not
SAVE = 0;                           % =1 save TRAIN, =0 do not
IMSIZE = [24 24];                   % standard example size
POS_LIM = Inf;                      % default max positive examples
NEG_LIM = Inf;                      % default max negative examples
V = 0;                              % default verbose
INIT = 0;                           % collect positive data when init =1
Amu = 1; Asig = .025;               % aspect ratio mean and covariance
opt = 1;
LOW_VARIANCE_THRESH = .3;           % minimum variance of FP example

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
end



%% collect and process the POSITIVE CLASS images only when initializing
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

% P and N contain the positive and negative examples of the training set.
N = [ ];   


%%  RECOLLECT THE NEGATIVE CLASS
% we search for examples inside of a dataset of full images until we find
% enough negative examples that produce false positives to continue.

d = dir(path0);  d = d(3:length(d));
for n = 1:length(d);
    A{n} = imread([path0 d(n).name]);
end

while length(N) < NEG_LIM
    % 1. randomly select an image
    I = A{ceil(length(d)*rand(1))};

    % 2. select an example size window randomly
    Isize = size(I);  aspect_ratio = gsamp(Amu, Asig, 1);  
    Wmu = IMSIZE(2); %Isize(2)/10 + IMSIZE(2);  %Wmu = IMSIZE(2) + (Isize(2) - IMSIZE(2))/2; 
    Wsig = Isize(2)*1.25;  W = 0;
    while (W < IMSIZE(2)) || (W > Isize(2))
        W = gsamp(Wmu, Wsig,1);
    end
    
    % 3. adjust it if necessary
    if ~isa(I, 'double')
        cls = class(I);
        I = mat2gray(I, [0 double(intmax(cls))]); 
    end
    if size(I,3) > 1
        I = rgb2gray(I);
    end
    
    % 4. select a window location and aspect ratio.  check for low 
    %    variance examples and reject 95% of them - they occur too often!
    EXAMPLE = zeros(IMSIZE);
    while var(EXAMPLE(:)) < LOW_VARIANCE_THRESH
        W = round(W);
        H = round(aspect_ratio*W);

        X = ceil(  (Isize(2) - W)  *rand(1));
        Y = ceil(  (Isize(1) - H) * rand(1));
        rect = [X Y W-1 H-1];

        EXAMPLE = imcrop(I, rect);

        % 5. normalize if necessary and resize to IMSIZE
        if var(EXAMPLE(:)) < LOW_VARIANCE_THRESH
            if rand > .95
                %disp('boring, but we keep it anyway');
                break;
            else
                %disp('too boring, find another!');
            end
        end
    end
    
    
    % 6. normalize if necessary and resize to IMSIZE
    EXAMPLE = imresize(EXAMPLE, IMSIZE);
    if NORM
        EXAMPLE = imnormalize('image', EXAMPLE);
    end
    II = integral_image(EXAMPLE);


    % 7. use the CASCADE to classify the randomly selected example, and keep it
    %    if it produces a false positive
    if INIT
        N(length(N)+1).Image = EXAMPLE; %#ok<AGROW>
        N(length(N)).II    = II(:); %#ok<AGROW>
        N(length(N)).class = 0; %#ok<AGROW>
    else
        if ada_classify_cascade(CASCADE, II, [0 0])      % = 1 is a false positive
            N(length(N)+1).Image = EXAMPLE; %#ok<AGROW>
            N(length(N)).II    = II(:); %#ok<AGROW>
            N(length(N)).class = 0; %#ok<AGROW>
        else
            %disp('keep searching.');
        end

    end
end

% merge the positive examples P and negative N into TRAIN
TRAIN = [P N];


% wrap it up by saving
if SAVE
    if V; disp(['saving ' FILENM ' ...']); end;
    save(FILENM, 'TRAINING');
end
