function [WEAK, PRE] = vj_find_haar_parameters2(h_ind, training_labels, PRE, WEAK, w)
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
err = zeros([1 length(training_labels)]);  %f = zeros(size(err)); 
polarity = zeros(size(err)); 

% look up and sort the feature responses for feature 'h_ind' over the training set
%[f, PRE] = bigmatrix_get_row(PRE, h_ind);
f = PRE.get_rows([h_ind h_ind], 'nosave');      % retreive the feature responses
[fsorted, inds] = sort(f);                      % sorted ascending feature responses
lsorted = training_labels(inds);                % labels sorted according to above
wsorted = w(inds);                              % weights sorted according to above

% efficient way to find optimal threshold from page 6 of [Viola-Jones IJCV 2004]
TPOS = sum(training_labels.*w);                 % Total sum of class 1 example weights
TNEG = sum( (~training_labels).*w);             % Total sum of class 0 example weights
SPOS = 0;                                       % Running sum of class 1 example weights
SNEG = 0;                                       % Running sum of class 0 example weights


% loop through the sorted list, updating SPOS and SNEG, and computing the ERROR
spostemp = zeros(size(w));
snegtemp = zeros(size(w));
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
[WEAK.minerr(h_ind), q_ind]   = min(err);
WEAK.theta(h_ind)           = fsorted(q_ind);
WEAK.polarity(h_ind)        = polarity(q_ind);


% %%%%%%%%%%%%%%%%%% DEBUG %%%%%%%%%%%%%%%%%5
% %stuck1 = 84039; 
% stuck1 = 0;
% stuck2 = 6851;
% if (h_ind == stuck2 ) || (h_ind == stuck1)
% 
%     if h_ind == stuck1; stuck = stuck1; end
%     if h_ind == stuck2; stuck = stuck2; end
%     
%     disp(['selected q_ind=' num2str(q_ind)]);
%     figure; 
%     subplot(2,3,1);
%     bar(fsorted); xlabel('Sorted Training Example Indexes'); xlim([1 length(training_labels)]); ylabel('Feature Response'); title(['Sorted Responses For Haar-Like Feature #' num2str(stuck) ' on Training Set']);
%     line([q_ind q_ind], [-0.6 0.6], 'LineWidth', 2, 'Color', [.8 0 0], 'LineStyle', '--');
%     subplot(2,3,2);
%     bar(inds); xlabel('Sorted Training Example Indexes'); xlim([1 length(training_labels)]); ylabel('Training Label Index'); title(['Example Indexes Sorted by For Haar-Like Feature Response #' num2str(stuck)]);
%     line([q_ind q_ind], [0 length(training_labels)], 'LineWidth', 2, 'Color', [.8 0 0], 'LineStyle', '--');
%     subplot(2,3,3);
%     bar(wsorted);  xlabel('Sorted Training Example Indexes'); xlim([1 length(training_labels)]); ylabel('Weight'); title(['Example Weights Sorted By Feature Response #' num2str(stuck)]);
%     line([q_ind q_ind], [0 max(wsorted)], 'LineWidth', 2, 'Color', [.8 0 0], 'LineStyle', '--');
%     subplot(2,3,4);
%     bar(polarity);  xlabel('Sorted Training Example Indexes'); xlim([1 length(training_labels)]); ylabel('Polarity'); title('Weak Classifier Search For Polarity');
%     line([q_ind q_ind], [-1 1], 'LineWidth', 2, 'Color', [.8 0 0], 'LineStyle', '--');
%     subplot(2,3,5);
%     bar(err); xlabel('Sorted Training Example Indexes'); xlim([1 length(training_labels)]); ylabel('Weak Classifier Error'); title('Search for Weak Classifier Threshold, Theta');
%     line([q_ind q_ind], [0 0.6], 'LineWidth', 2, 'Color', [.8 0 0], 'LineStyle', '--');
%     %subplot(2,3,6);
%     %plot(spostemp, 'b-'); hold on; plot(snegtemp, 'r-');xlim([1 length(training_labels)]);
%     
%     keyboard;
% end

