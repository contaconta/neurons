function [error, theta, pol] = ada_spedge_learn(i, training_labels, SET, w)
%
%
%
%
%
%
%


%f = ada_spedge_response(WEAK.(field)(i).index, PRE);
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

%% find 'q' that gives the minimum error and correspoinding polarity
[error, q_ind]  = min(err);
theta           = fsorted(q_ind);
pol             = polarity(q_ind);





%% a more simple version but slightly slower

% %% get the unique spedge responses to the training data
% f = ada_spedge_response(WEAK.(field)(i).descriptor, TRAIN);
% funique = unique(f);
% 
% % initialize some variables to make loops faster by preallocating memory
% err = zeros(size(funique)); 
% polarity = zeros(size(err)); 
% training_labels = [TRAIN(:).class];
% 
% 
% for q = 1:length(funique);
%     
%     posabove = (f >= funique(q))  & (training_labels == 1);
%     negbelow = (f < funique(q)) & (training_labels == 0);
%     
%     err(q) = sum(w(posabove)) + sum(w(negbelow));
%     
%     if err(q) > .5
%         err(q) = 1 - err(q);
%         polarity(q) = -1;
%     else
%         polarity(q) = 1;
%     end
% 
% end
% 
% %% find 'q' that gives the minimum error and correspoinding polarity
% [error, q_ind]  = min(err);
% theta           = funique(q_ind);
% pol             = polarity(q_ind);
% 
% 
% 





%% %%%%%%%%%%%%%%%%%% DEBUG GRAPHICS %%%%%%%%%%%%%%%%%5
% %stuck1 = 84039; 
% stuck1 = 0;
% stuck2 = 1;
% if (i == stuck2 ) || (i == stuck1)
% 
%     if i == stuck1; stuck = stuck1; end
%     if i == stuck2; stuck = stuck2; end
%     
%     disp(['selected q_ind=' num2str(q_ind)]);
%     figure; 
%     subplot(2,3,1);
%     bar(fsorted); xlabel('Sorted Training Example Indexes'); xlim([1 length(training_labels)]); ylabel('Feature Response'); title(['Sorted Responses For Haar-Like Feature #' num2str(stuck) ' on Training Set']);
%     line([q_ind q_ind], [0 max(f)], 'LineWidth', 2, 'Color', [.8 0 0], 'LineStyle', '--');
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
% %     plot(spostemp, 'b-'); hold on; plot(snegtemp, 'r-');xlim([1 length(training_labels)]);
%     
%     keyboard;
% end
