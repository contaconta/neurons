function [WEAK, PRE] = vj_fast_find_haar_parameters(h_ind, training_labels, PRE, WEAK, w)
%VJ_FIND_HAAR_PARAMETERS deteremines WEAK classifier threshold, min error, polarity
%
%   [TRAIN, PRE] = vj_find_haar_parameters(h_ind, TRAIN, train_list, PRE, WEAK, w)
%   uses training examples from the structure in TRAIN to 
%   deteremine the optimal threshold and polarity (minimizing classification 
%   error) for the WEAK classifier specified by 'h_ind'.  Also requires the 
%   weight vector, 'w'.
%
%
%   Copyright © 2008 Kevin Smith
%   See also VJ_TRAIN, VJ_ADABOOST, VJ_DEFINE_CLASSIFIERS, BIGMATRIX_GET_ROW

blocksize = 1;
blockstart = 1;  blockend = blocksize + blockstart - 1;  %blockend = PRE.block_size;

tic;
while blockend <= size(WEAK.descriptor,1)
    
    %disp(['sorting blockstart ' num2str(blockstart) ' to blockend ' num2str(blockend) ]);
   
    f = PRE.data.f_responses(blockstart:blockend,:);
    
    blockstart = blockend + 1;
    blockend = blockstart + blocksize -1;
    
    %disp([' sorting ' ]);
    [f, inds] = sort(f,2);
    
    
    %training_labels = repmat(training_labels, [ size(f,1) 1]);
    %w = repmat(w, [size(f,1) 1]);
    
    
end
toc