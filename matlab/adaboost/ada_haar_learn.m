function  [minerr, theta, pol] = ada_haar_learn(h_ind, training_labels, SET, w)
%ADA_HAAR_LEARN finds parameters for optimal class separation
%
%   [TRAIN, PRE] = ada_haar_learn(h_ind, training_labels, SET, w)
%   uses training examples from the structure in TRAIN to 
%   deteremine the optimal threshold and polarity (minimizing classification 
%   error) for the WEAK classifier specified by 'h_ind'.  Also requires the 
%   weight vector, 'w'.
%
%
%   Copyright © 2008 Kevin Smith
%   See also ADA_ADABOOST


% initialize some variables to make loops faster by preallocating memory
err = zeros([1 length(training_labels)]);  %f = zeros(size(err)); 
polarity = zeros(size(err)); 

% look up and sort the feature responses for feature 'h_ind' over the training set
f = SET.responses.getCols(h_ind);               % retreive responses of feature h_ind to training set
[fsorted, inds] = sort(f);                      % sorted ascending feature responses
lsorted = training_labels(inds);                % labels sorted according to above
wsorted = w(inds);                              % weights sorted according to above

% efficient way to find optimal threshold from page 6 of [Viola-Jones IJCV 2004]
TPOS = sum(training_labels.*w);                 % Total sum of class 1 example weights
TNEG = sum( (~training_labels).*w);             % Total sum of class 0 example weights
SPOS = 0;                                       % Running sum of class 1 example weights
SNEG = 0;                                       % Running sum of class 0 example weights


% loop through the sorted list, updating SPOS and SNEG, and computing the ERROR
for q=1:length(fsorted)
    
    % if your feature has the same value for many examples in the training
    % set, this can cause a problem when selecting a threshold if you keep 
    % accumulating the error (within a block of repeated values).  For a
    % given feature response, there should be only 1 error value!
    if (q>1) && (fsorted(q) == fsorted(q-1))
        err(q) = err(q-1); polarity(q) = polarity(q-1);
    else
        
        % compute the classification error if we set the treshold to 'q'
        if SPOS + (TNEG - SNEG) <= SNEG + (TPOS - SPOS)
            err(q) = SPOS + (TNEG - SNEG); polarity(q) = -1;     % The polarity = -1 case
        else
            err(q) = SNEG + (TPOS - SPOS); polarity(q) = 1;      % The polarity = +1 case
        end
    end
        
    % update SPOS and SNEG for the next iteration
    if lsorted(q) == 1
        SPOS = SPOS + wsorted(q);
    else
        SNEG = SNEG + wsorted(q);
    end
    
    
end

% find 'q' that gives the minimum error and correspoinding polarity
[minerr, q_ind] = min(err);
theta           = fsorted(q_ind);
pol             = polarity(q_ind);




