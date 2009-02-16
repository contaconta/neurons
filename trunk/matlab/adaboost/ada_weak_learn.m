function [error, theta, pol] = ada_weak_learn(i, training_labels, SET, w)


f = SET.responses.getCols(i);                   % retreive responses of feature i to training set
err = zeros(size(f)); 
polarity = zeros(size(err)); 

%% sort the responses, and similarly sort the labels and weights
[fsorted, inds] = sort(f);                      % sorted ascending feature responses
lsorted = training_labels(inds);                % labels sorted according to above
wsorted = w(inds);                              % weights sorted according to above


%% efficient way to find optimal threshold from page 6 of [Viola-Jones IJCV 2004]
TPOS = sum(training_labels.*w);                 % Total sum of class 1 example weights
TNEG = sum( (~training_labels).*w);             % Total sum of class 0 example weights
SPOS = 0;                                       % Running sum of class 1 example weights
SNEG = 0;                                       % Running sum of class 0 example weights

%% loop through the sorted list, updating SPOS and SNEG, and computing the ERROR
for q=1:length(fsorted)
    
    % if your feature has the same value for many examples in the training
    % set, this can cause a problem when selecting a threshold if you keep 
    % accumulating the error (within a block of repeated values).  For a
    % given feature response, there should be only 1 error value!
    if (q>1) && (fsorted(q) == fsorted(q-1))
        err(q) = err(q-1); polarity(q) = polarity(q-1);
    else
        % compute the classification error if we set the threshold to 'q'
        % keep in mind, SPOS is sum up to but not including q, as is SNEG
%         fp_error_pos_pol = SPOS + lsorted(q)*wsorted(q);            % fp >= q
%         fp_error_neg_pol = TPOS - SPOS;                             % fp <= q
%         fn_error_pos_pol = TNEG - SNEG + ~lsorted(q)*wsorted(q);    % fn > q
%         fn_error_neg_pol = SNEG;                                    % fn < q
        
        fp_error_pos_pol = TPOS - SPOS;                             % fp >= q
        fp_error_neg_pol = SPOS + lsorted(q)*wsorted(q);            % fp <= q
        fn_error_pos_pol = SNEG;                                    % fn < q
        fn_error_neg_pol = TNEG - SNEG - ~lsorted(q)*wsorted(q);    % fn > q
        
%         fp_error_pos_pol = SPOS;            % fp >= q
%         fp_error_neg_pol = TPOS - SPOS;                             % fp <= q
%         fn_error_pos_pol = TNEG - SNEG;                                    % fn < q
%         fn_error_neg_pol = SNEG;    % fn > q
        
        pos_pol_error = fp_error_pos_pol + fn_error_pos_pol;
        neg_pol_error = fp_error_neg_pol + fn_error_neg_pol;
        
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
theta           = fsorted(q_ind);
pol             = polarity(q_ind);

%keyboard;
