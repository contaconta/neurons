function PRE = vj_precompute_haar_response_new(TRAIN, WEAK, filenm, matpath,PRE)
%VJ_PRECOMPUTE_HAAR_RESPONSE precomputes and stores haar feature responses
%
%   m = vj_precompute_haar_response(TRAIN, WEAK, filenm) takes a struct TRAIN
%   containing the training data and a struct WEAK containing the weak
%   haar-like features and precomputes the responses of each feature to
%   each piece of training data.  The results are stored in PRE: haar
%   features in the rows and training data in the columns.  Optional
%   argument filenm allows you to chose the filename to store to.
%
%   Example:  To get retreive a row vector containing the feature responses
%   for feature F over the whole training set, use
%   ------------------------------------------------------------------
%   [f_response, PRE] = bigmatrix_get_row(PRE, feature_index);
%
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT, VJ_TRAIN, INTEGRAL_IMAGE


%-----------------------------------------------------------
% define a path and filename for our memory map file
%-----------------------------------------------------------
if nargin < 3
    filenm = 'MATLAB_MEMMAP_';
    matpath = [pwd '/mat/'];
end
BYTESIZE = 250000000;

if isempty(PRE)
    % initialize the bigarray for the first go
    PRE = bigarray(size(WEAK.descriptor,1), length(TRAIN), 'filename', filenm, 'bytes', BYTESIZE, 'path', matpath, 'type', 'matlab .mat file');
    %PRE = bigarray(size(WEAK.descriptor,1), length(TRAIN), 'filename', filenm, 'bytes', BYTESIZE, 'path', matpath, 'type', 'memmapfile');
end

%--------------------------------------------------------------------------
% precompute the haar-like feature responses of each feature to each
% training image.  
%--------------------------------------------------------------------------

block = round(BYTESIZE / (length(TRAIN)*8)); %1000;
IIs = [TRAIN(:).II];                        % vectorized integral images
f_responses = zeros(block,length(TRAIN));   % preallocated haar response matrix
W = wristwatch('start', 'end', size(WEAK.descriptor,1), 'every', 10000, 'text', '    ...precomputed haar ');
j = 1;

for i = 1:size(WEAK.descriptor,1)
    
    f_responses(j,:) = vj_fast_haar_response(WEAK.fast(i,:), IIs);
    
    if mod(i,block) == 0
        PRE.store_rows(f_responses, [i-block+1 i]);
        %m.data.f_responses(i-block+1:i,:) = f_responses;
        j = 0;
    end
    W = wristwatch(W, 'update', i);
    j = j + 1;
end
PRE.store_rows(f_responses, [i-j+2 i]);
PRE.force_save;
%m.data.f_responses(i-j+2:i,:) = f_responses(1:j-1,:);

