function temp_montage(varargin)
%TEMP_MONTAGE creates a montage of training or validation examples
%
%   temp_montage(TRAIN, ...) is called with the following optional 
%                            arguments (TRAIN can be a TRAIN or VALIDATION
%                            structure.
%
%       'random', NUM                   % selects NUM indexes at random
%       'inds', index_list              % you specifiy the index_list
%       'class', 0 or 1 or 'both'       % if random, include class 1 or 0 or both
%


%% handle the arguments
TRAIN = varargin{1};
RAND = 1;  CLASS = 1;  num = 400;
for i = 2:nargin 
    if strcmp('random', varargin{i})
        RAND = 1;
        num = varargin{i+1};
    end
    if strcmp('inds', varargin{i})
        RAND = 0;
        num = length(varargin{i+1});
        inds = varargin{i+1};
    end
    if strcmp('class', varargin{i})
        CLASS = varargin{i+1};
    end
end

%% if necessary, select random image indices
if RAND == 1
    if strcmp(CLASS, 'both')
        inds = 1:length(TRAIN);
    else
        [vals,inds] = find([TRAIN.class] == CLASS);        
    end
    
end

%% allocate the montage
IMSIZE = size(TRAIN.Images(:,:,1));
I = zeros([IMSIZE(1) IMSIZE(2) 1 100]);


%% randomly select images
if RAND == 1
    for j = 1:num;  a(j) = inds(ceil(length(inds)*rand(1)));  inds = setdiff(inds, a(j)); end;
    a = sort(a);
else
    a = inds;
end

%% collect images

c = 1;
for j = a

    I(:,:,1,c) = TRAIN.Images(:,:,a(c));

    c = c + 1;
end
figure; 
montage(I);
title([num2str(length(a)) ' examples of ' num2str(CLASS) ' class(es).']);
    
    