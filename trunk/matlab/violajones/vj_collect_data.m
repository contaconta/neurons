function TRAIN = vj_collect_data(varargin)
%VJ_COLLECT_DATA organizes training images for viola-jones
%
%   TRAIN = vj_collect_data(path1, path0, ...) collects and processes 
%   training images found in path1 (positive training class) and path0
%   (negative training class).  The data is stored into struct TRAIN for
%   use in other vj modules.  Optional argument 'SAVE' will save TRAIN to
%   a specified filename.  Optional argument 'IMRESIZE' allows the user to
%   define the training image size [WIDTH HEIGHT] (all images will be 
%   resized to a default size of [24 24]).  Optional argument 'normalize' 
%   toggles if the image histogram should be normalized (1=yes/0=no).
%   Optional argument 'DATA_LIMIT' will select a subset of the first
%   [POS_EXAMPLES NEG_EXAMPLES] from the training set.  Optional argument
%   'v' for verbose output.
%
%   Example:
%   collects 100 + example images from path1 and 100 - example images from path0,
%   resized them to [12 12], does not normalize them, and save them to
%   TRAIN.mat and returns a struct TRAIN containing the training data.
%   --------------
% 
%   TRAIN = vj_collect_data(path1, path0, 'size', [12 12], 'save', ...
%           'TRAIN.mat', 'normalize', 0, 'data_limit', [100 100]);
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, VJ_TRAIN, INTEGRAL_IMAGE, VJ_ADABOOST
 


% define parameters
path1 = varargin{1}; 
path0 = varargin{2}; 
NORM = 1; 
SAVE = 0; 
IM_SIZE = [24 24];
POS_LIM = Inf;
NEG_LIM = Inf;
V = 0;
TRAIN = [];
 
% handle optional arguments
if nargin > 2;
    for i = 3:nargin
        if strcmp(varargin{i}, 'size')
            IM_SIZE = varargin{i+1};
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
    end
end

% collect and process the POSITIVE CLASS images
d = dir(path1); count = 1;

for i = 1:length(d)
    if count <= POS_LIM
        filenm = [path1 d(i).name];
        try 
            I = imread(filenm);
        catch %#ok<CTCH>
            continue
        end
        
        if V; disp(['reading ' filenm '...']); end;
        
        if ~isa(I, 'double')
            cls = class(I);
            I = mat2gray(I, [0 double(intmax(cls))]); 
        end
        if size(I,3) > 1
            I = rgb2gray(I);
        end
        if ~isequal(size(I), IM_SIZE)
            I = imresize(I, IM_SIZE);
        end
        if NORM
            I = imnormalize('image', I);
        end
        II = integral_image(I);

        TRAIN(length(TRAIN)+1).Image = I; %#ok<AGROW>
        TRAIN(length(TRAIN)).II    = II(:); %#ok<AGROW>
        TRAIN(length(TRAIN)).class = 1; %#ok<AGROW>
        count = count + 1;       
    end
end

% collect and process the NEGATIVE CLASS images
d = dir(path0);  count = 1;

for i = 1:length(d)
    if count <= NEG_LIM
        filenm = [path0 d(i).name];
        try 
            I = imread(filenm);
        catch %#ok<CTCH>
            continue
        end
        
        if V; disp(['reading ' filenm '...']); end;
        
        if ~isa(I, 'double')
            cls = class(I);
            I = mat2gray(I, [0 double(intmax(cls))]); 
        end
        if size(I,3) > 1
            I = rgb2gray(I);
        end
        if ~isequal(size(I), IM_SIZE)
            I = imresize(I, IM_SIZE);
        end
        if NORM
            I = imnormalize('image', I);
        end
        II = integral_image(I);

        TRAIN(length(TRAIN)+1).Image = I; %#ok<AGROW>
        TRAIN(length(TRAIN)).II    = II(:); %#ok<AGROW>
        TRAIN(length(TRAIN)).class = 0; %#ok<AGROW>
        count = count + 1;       
    end
end


% wrap it up by saving
if SAVE
    if V; disp(['saving ' FILENM ' ...']); end;
    save(FILENM, 'TRAIN');
end
 