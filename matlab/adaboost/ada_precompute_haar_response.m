function PRE = ada_precompute_haar_response(TRAIN, WEAK, filenm, matpath,PRE)
%ADA_PRECOMPUTE_HAAR_RESPONSE precomputes and stores haar feature responses
%
%   m = ada_precompute_haar_response(TRAIN, WEAK, filenm) takes a struct TRAIN
%   containing the training data and a struct WEAK containing the weak
%   haar-like features and precomputes the responses of each feature to
%   each piece of training data.  The results are stored in PRE: haar
%   features in the rows and training data in the columns.  Optional
%   argument filenm allows you to chose the filename to store to.
%
%   Example:  To get retreive a row vector containing the feature responses
%   for feature F over the whole training set, use
%   ------------------------------------------------------------------
%
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, ADA_TRAIN_CASCADE, INTEGRAL_IMAGE


%-----------------------------------------------------------
% define a path and filename for our memory map file
%-----------------------------------------------------------
if nargin < 3
    filenm = 'MATLAB_MEMMAP_';
    matpath = [pwd '/mat/'];
end
BYTESIZE = 250000000;

% get a list of haar-like features
haar_list = [];
for i = 1:length(WEAK.learners)
    if strcmp('haar', WEAK.learners{i}{1})
        haar_list = [haar_list WEAK.learners{i}{3}];
    end
end

if isempty(PRE)
    % initialize the bigarray for the first go
    PRE = bigarray(length(haar_list), length(TRAIN), 'filename', filenm, 'bytes', BYTESIZE, 'path', matpath, 'type', 'matlab .mat file');
end

%--------------------------------------------------------------------------
% precompute the haar-like feature responses of each feature to each
% training image.  
%--------------------------------------------------------------------------

block = round(BYTESIZE / (length(TRAIN)*8)); %1000;
IIs = [TRAIN(:).II];                        % vectorized integral images
f_responses = zeros(block,length(TRAIN));   % preallocated haar response matrix
W = wristwatch('start', 'end', length(haar_list), 'every', 10000, 'text', '    ...precomputed haar ');
j = 1;


for i = haar_list
    field = WEAK.ptr{i,1};
    ind = WEAK.ptr{i,2};
    f_responses(j,:) = ada_haar_response(WEAK.(field)(ind).hinds, WEAK.(field)(ind).hvals, IIs);
    
    if mod(i,block) == 0
        PRE.store_rows(f_responses, [i-block+1 i]);
        j = 0;
    end
    W = wristwatch(W, 'update', i);
    j = j + 1;
end
PRE.store_rows(f_responses, [i-j+2 i]);
PRE.force_save;


