function [error, threshold, pol] = p_select_weak_parameters(learner_id, learner_data, SET, w, l)
%% TODO: write documenation

%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%               Aurelien Lucchi     http://cvlab.epfl.ch/~lucchi/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

%responses = p_dummy_feature_values(learner_id, SET);
%responses = double(p_get_feature_responses(SET, {learner_id}, l));
responses = p_get_feature_responses(SET, learner_id, learner_data, l);

%keyboard;

err = zeros(size(responses)); 
polarity = zeros(size(err)); 

%% sort the responses, and similarly sort the labels and weights
[fsorted, inds] = sort(responses);              % sorted ascending feature responses
lsorted = SET.class(inds);                      % labels sorted according to above
wsorted = w(inds);                              % weights sorted according to above


%% efficient way to find optimal threshold from page 6 of [Viola-Jones IJCV 2004]
TPOS = sum((SET.class==1).*w);                  % Total sum of class 1 example weights
TNEG = sum((SET.class==-1).*w);                 % Total sum of class -1 example weights
SPOS = 0;                                       % Running sum of class 1 example weights
SNEG = 0;                                       % Running sum of class -1 example weights

%% loop through the sorted list, updating SPOS and SNEG, and computing the ERROR
for q=1:length(fsorted)
    
    % if your feature has the same value for many examples in the training
    % set, this can cause a problem when selecting a threshold if you keep 
    % accumulating the error (within a block of repeated values).  For a
    % given feature response, there should be only 1 error value!
    if (q>1) && (fsorted(q) == fsorted(q-1))
        err(q) = err(q-1); polarity(q) = polarity(q-1);
    else
        % accumulate error for positive polarity and negative polarity.
        % errors come in the form of false positives (fp) and false
        % negatives (fn)
        
        fp_error_pos_pol = TPOS - SPOS;                             % fp >= q
        fp_error_neg_pol = SPOS + lsorted(q)*wsorted(q);            % fp <= q
        fn_error_pos_pol = SNEG;                                    % fn < q
        fn_error_neg_pol = TNEG - SNEG - ~lsorted(q)*wsorted(q);    % fn > q

        % total error for +/- polarity
        pos_pol_error = fp_error_pos_pol + fn_error_pos_pol;
        neg_pol_error = fp_error_neg_pol + fn_error_neg_pol;
        
        % set the final error and polarity
        if pos_pol_error <= neg_pol_error
            err(q) = pos_pol_error; polarity(q) = 1;
        else
            err(q) = neg_pol_error; polarity(q) = -1;
        end
    end
    
    % update SPOS and SNEG for the next iteration
    if lsorted(q) == 1
        SPOS = SPOS + wsorted(q);
    else
        SNEG = SNEG + wsorted(q);
    end
    
end

%% find 'q' that gives the minimum error and correspoinding polarity
[error, q_ind]  = min(err);
threshold       = double(fsorted(q_ind));
pol             = polarity(q_ind);

%keyboard;
