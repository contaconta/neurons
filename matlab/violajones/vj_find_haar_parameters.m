function [WEAK, PRE] = vj_find_haar_parameters(h_ind, TRAIN, PRE, WEAK, w)
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


% initialize some variables to make loops faster by preallocating memory
err = zeros([1 length(TRAIN)]);  f = zeros(size(err)); polarity = zeros(size(err)); %#ok<NASGU>

% form a vector containing class labels for the training data
training_labels = [TRAIN(:).class];


% look up and sort the feature responses for feature 'h_ind' over the training set
[f, PRE] = bigmatrix_get_row(PRE, h_ind);
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
    
    % compute the classification error if we set the treshold to 'q'
    if SPOS + (TNEG - SNEG) <= SNEG + (TPOS - SPOS)
        err(q) = SPOS + (TNEG - SNEG); polarity(q) = -1;     % The polarity = -1 case
    else
        err(q) = SNEG + (TPOS - SPOS); polarity(q) = 1;      % The polarity = +1 case
    end
    
    % update SPOS and SNEG for the next iteration
    if lsorted(q) == 1
        SPOS = SPOS + wsorted(q);
    else
        SNEG = SNEG + wsorted(q);
    end
end

% find 'q' that gives the minimum error and correspoinding polarity
[WEAK.minerr(h_ind), ind]   = min(err);
WEAK.theta(h_ind)           = fsorted(ind);
WEAK.polarity(h_ind)        = polarity(ind);

%keyboard;
