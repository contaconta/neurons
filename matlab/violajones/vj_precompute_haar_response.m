function PRE = vj_precompute_haar_response(TRAIN, WEAK, filenm, matpath)
%VJ_PRECOMPUTE_HAAR_RESPONSE precomputes and stores haar feature responses
%
%   PRE = vj_precompute_haar_response(TRAIN, WEAK, filenm) takes a struct TRAIN
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


if nargin < 3
    filenm = 'BIGMATRIX_';
    matpath = [pwd '/mat/'];
end

f_responses = zeros(1,length(TRAIN));
PRE = bigmatrix_store_rows(1, f_responses, 'path', matpath, 'filenm', filenm);
block = size(PRE.data, 1);
f_responses = zeros(block,length(TRAIN));

IIs = [TRAIN(:).II];

j=1;
W = wristwatch('start', 'end', size(WEAK.descriptor,1), 'every', 10000, 'text', '    ...precomputed haar ');
for i = 1:size(WEAK.descriptor,1)
   
    %frep = repmat(WEAK.fast(i,:)', [1 length(TRAIN)]);
    f_responses(j,:) = vj_fast_haar_response(WEAK.fast(i,:), IIs);
    
    if mod(i, block) == 0
        PRE = bigmatrix_store_rows(i-block+1:i, f_responses, PRE,'v');
        j = 0;
    end
    W = wristwatch(W, 'update', i);
    j = j + 1;
end
PRE = bigmatrix_store_rows(i-j+2:i, f_responses(1:j-1,:), PRE,'v');
bigmatrix_save(PRE, 'v');
