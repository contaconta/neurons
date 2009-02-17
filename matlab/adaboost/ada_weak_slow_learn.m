function [error, theta, pol] = ada_weak_slow_learn(i, training_labels, SET, w)


f = SET.responses.getCols(i);                   % retreive responses of feature i to training set
%err = zeros(size(f)); 
err = ones(size(f)); 
polarity = zeros(size(err)); 

%% sort the responses, and similarly sort the labels and weights
[fsorted, inds] = sort(f);                      % sorted ascending feature responses
lsorted = training_labels(inds);                % labels sorted according to above
wsorted = w(inds);                              % weights sorted according to above

q = 1;

while q <= length(fsorted)
    
    % polarity +1 case - FPs should be >= q, FNs should be < q
    FPerr = sum(wsorted(q:length(fsorted)) .* lsorted(q:length(fsorted)));
    if q > 1
        FNerr = sum(wsorted(1:q-1) .* ~lsorted(1:q-1));
    else
        FNerr = 0;
    end
    pospolarityerror = FPerr + FNerr;
    
    % polarity -1 case - FPs should be <= q, FNs should be > q
    FPerr = sum(wsorted(1:q) .* lsorted(1:q));
    if q < length(fsorted)
        FNerr = sum(wsorted(q+1:length(fsorted)) .* ~lsorted(q+1:length(fsorted)));
    else
        FNerr = 0;
    end
    
    negpolarityerror = FPerr + FNerr;

    if pospolarityerror <= negpolarityerror
        err(q) = pospolarityerror;
        polarity(q) = 1;
    else
        err(q) = negpolarityerror;
        polarity(q) = -1;
    end
    
    % set all the features with same value to have same error
    err( fsorted == fsorted(q)) = err(q);
    
    q = find(fsorted > fsorted(q),1);
end




%% find 'q' that gives the minimum error and correspoinding polarity
[error, q_ind]  = min(err);
theta           = fsorted(q_ind);
pol             = polarity(q_ind);

% esorted = sort(err);
% disp(['feature ' num2str(i) ', min errors = ' num2str(esorted(1:5)')]);

%====================debugging ===========================
% learner.polarity = pol;
% learner.theta = theta;
% h = ada_classify_weak_learner(i, learner, SET)';
% hsorted = h(inds);
% if pol == 1
%     hhand(1:q_ind-1) = ~lsorted(1:q_ind-1);
%     hhand(q_ind:length(fsorted)) = lsorted(q_ind:length(fsorted));
% else
%     hhand(1:q_ind) = ~lsorted(1:q_ind-1);
%     hhand(q_ind+1:length(fsorted)) = lsorted(q_ind:length(fsorted));
% end

% keyboard;




%keyboard;

% %% efficient way to find optimal threshold from page 6 of [Viola-Jones IJCV 2004]
% TPOS = sum(training_labels.*w);                 % Total sum of class 1 example weights
% TNEG = sum( (~training_labels).*w);             % Total sum of class 0 example weights
% SPOS = 0;                                       % Running sum of class 1 example weights
% SNEG = 0;                                       % Running sum of class 0 example weights
% 
% %% loop through the sorted list, updating SPOS and SNEG, and computing the ERROR
% for q=1:length(fsorted)
%     
%     % if your feature has the same value for many examples in the training
%     % set, this can cause a problem when selecting a threshold if you keep 
%     % accumulating the error (within a block of repeated values).  For a
%     % given feature response, there should be only 1 error value!
%     if (q>1) && (fsorted(q) == fsorted(q-1))
%         err(q) = err(q-1); polarity(q) = polarity(q-1);
%     else
%         
%         % compute the classification error if we set the treshold to 'q'
%         if SPOS + (TNEG - SNEG) <= SNEG + (TPOS - SPOS)
%             err(q) = SPOS + (TNEG - SNEG); polarity(q) = -1;     % The polarity = -1 case
%         else
%             err(q) = SNEG + (TPOS - SPOS); polarity(q) = 1;      % The polarity = +1 case
%         end
%     end
%     
%     % update SPOS and SNEG for the next iteration
%     if lsorted(q) == 1
%         SPOS = SPOS + wsorted(q);
%     else
%         SNEG = SNEG + wsorted(q);
%     end
%     
% end
