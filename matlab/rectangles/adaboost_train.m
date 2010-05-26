%% LOAD PARAMETERS
adaboost_settings;


%% PRE-BOOSTING

% pre-generate the set of features
if VJ == 1
    [R,C,N,P] = generate_viola_jones_features(IMSIZE);
    %[R,C,N,P] = generate_viola_jones_features(IMSIZE, 'shapes', {'horz2', 'vert2'});
    %[R,C,N,P] = generate_viola_jones_features(IMSIZE, 'shapes', {'horz3', 'vert3'});
end

if ~exist('D.mat', 'file')
    % define the training data set
    [Lp,Dp] = collect_positive_examples(N_pos, IMSIZE, pos_train_folder); N_pos = length(Lp);

    % define the test data set
    [Ln,Dn] = collect_negative_examples(N_total-N_pos, IMSIZE, neg_train_folder);

    D = [Dp;Dn];  clear Dp Dn;  % D contains all integral image data (each row contains a vectorized image)
    L = [Lp;Ln];  clear Lp Ln;  % L contains all associated labels
    save('D.mat', '-v7.3', 'D', 'L');  disp(['...storing ' num2str(sum(L==1)) ' (class +1) / ' num2str(sum(L==-1)) ' (class -1) examples to D.mat.']);
else
    disp('...loading training data from D.mat');
    load D.mat;
end

% initialize the weights, set each class to have equal weights initially
W = ones(size(L));          % example weights
W(L == 1) = .5 * (W(L == 1) / sum(W(L == 1)));
W(L == -1) = .5 * (W(L == -1) / sum(W(L == -1)));


%% PERFORM BOOSTING

stats = zeros(T,7);     % keep statistics for each boosting round
error = zeros(T,1);     % weighted classification error at each boosting step
beta = zeros(T,1);      % computed beta value at each boosting step
alpha = zeros(T,1);     % computed alpha value at each boosting step
CLASSIFIER.rects = {}; CLASSIFIER.thresh = []; CLASSIFIER.pols = {}; CLASSIFIER.tpol = []; CLASSIFIER.alpha = [];

for t = 1:T
    disp(' ');
    disp('===================================================================');
    disp(['                   BOOSTING ROUND t = ' num2str(t) ]);
    disp('-------------------------------------------------------------------');
    
    % randomly sample features for this round of boosting
    if VJ == 1
        inds = randsample(size(N,1), N_features);
        f_rects = N(inds);  % randomly selected rectangles
        f_pols = P(inds);   % associated polarities
    else
        disp(['...generating Rank [2 to ' num2str(RANK) '] rectangles.']);
        %[tempr, tempc, f_rects, f_pols] = generate_rectangles(N_features, IMSIZE, RANK);
        [tempr, tempc, f_rects, f_pols] = generate_rectangles2(N_features, IMSIZE, RANK);
        clear tempr tempc;
    end
    
%     %%% TEMPORARY VISUALIZATION
%     figure(34334); disp('   VISUALIZING FEATURES');
%     for i = 1:N_features
%         rect_vis_ind(zeros(IMSIZE), f_rects{i}, f_pols{i});
%     end
    
        
    % populate the feature responses for the sampled features
    disp('...computing feature responses for the selected features.');
    F = zeros(size(D,1), size(f_rects,1), 'single'); tic;
    for i = 1:N_features
        %rect_vis_ind(zeros(IMSIZE), f_rects{i}, f_pols{i});
        F(:,i) = haar_feature(D, f_rects{i}, f_pols{i});
    end
    to=toc;  disp(['   Elapsed time ' num2str(to) ' seconds.']);
    
    %% find the best weak learner
    disp('...selecting the best weak learner and its parameters.');
    
    % only use a subset of the whole training set to find optimal params
    pos_inds = find(L == 1); neg_inds = find(L == -1);
    neg_inds = weight_sample(neg_inds, W(neg_inds), N_SAMPLES);
    inds = [pos_inds; neg_inds];
    Fsub = F(inds,:);  Lsub = L(inds);     % Wsub = normalize_weights(W(inds));
    Wsub = W(inds); Wsub(Lsub == -1) = Wsub(Lsub==-1) * ( sum(W(L==-1))/ sum(Wsub(Lsub==-1)));
    
    tic; [thresh p e ind] = best_weak_learner(Wsub,Lsub,Fsub);  % subset of training data

    %tic; [thresh p e ind] = best_weak_learner(W,L,F);          % entire set of training data
    %rect_vis_ind(zeros(IMSIZE), f_rects{ind}, f_pols{ind}); 
    to = toc; disp(['...selected feature ' num2str(ind) ' thresh = ' num2str(thresh) '. Polarity = ' num2str(p) '. Elapsed time is ' num2str(to) ' seconds.']);

    
    %% compute error. sanity check: best weak learner should beat 50%
    E = p*F(:,ind) < p*thresh;              % predicted classes from weak learner ( < for positive polarity )
    E = single(E); E(E ==0) = -1;           
    %E = adaboost_classify(rects, pols, thresh, tpol, alpha, D);
    correct_classification = (E == L);
    incorrect_classification = (E ~= L);
    error(t) = sum(W .* incorrect_classification);
    disp(['...local weighted error = ' num2str(e) ' global weighted error = ' num2str(error(t))]);

    %% update the weights
    beta(t) = error(t) / (1 - error(t) );
    alpha(t) = log(1/beta(t));
    W = W .* ((beta(t)*ones(size(W))).^(1-incorrect_classification));
    W = normalize_weights(W);
    
    % add the new weak learner to the strong classifier
    CLASSIFIER.rects{t} = f_rects{ind}; 
    CLASSIFIER.pols{t} = f_pols{ind};
    CLASSIFIER.tpol(t) = p;
    CLASSIFIER.thresh(t) = thresh;
    CLASSIFIER.alpha(t) = alpha(t);
    
    % evaluate the strong classifier and record performance
%     tic;
%     PR = adaboost_classify(CLASSIFIER.rects, CLASSIFIER.pols, CLASSIFIER.thresh, CLASSIFIER.tpol, CLASSIFIER.alpha, D);
%     [TP TN FP FN TPR FPR ACC] = rocstats(PR>0,L>0, 'TP', 'TN', 'FP', 'FN', 'TPR', 'FPR', 'ACC');
%     stats(t,:) = [TP TN FP FN TPR FPR ACC]; to = toc;
%     disp(['   TP = ' num2str(TP) '/' num2str(sum(L==1)) '  FP = ' num2str(FP) '/' num2str(sum(L==-1)) '  ACC = ' num2str(ACC)  ' Elapsed time ' num2str(to) ' seconds.']);
        
    % store a temporary copy
    save([results_folder EXP_NAME '-' host '-' date '.mat'], 'CLASSIFIER', 'W', 'stats', 'error');
    %save([results_folder EXP_NAME '-' host '-' date '.mat'], 'CLASSIFIER', 'W', 'error');
    
    % check for convergence (?)
    
    %keyboard;
end
