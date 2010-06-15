%% LOAD PARAMETERS
adaboost_settings;


%% PRE-BOOSTING

% pre-generate necessary feature pools (VJ, Lienhart, ...)
adaboost_pregenerate_features;

% load the database into D
adaboost_load_database;

% initialize the weights, set each class to have equal weights initially
W = ones(size(L));          % example weights
W(L == 1) = .5 * (W(L == 1) / sum(W(L == 1)));
W(L == -1) = .5 * (W(L == -1) / sum(W(L == -1)));


%% PERFORM BOOSTING

stats = zeros(T,7);     % keep statistics for each boosting round
error = zeros(T,1);     % weighted classification error at each boosting step
beta = zeros(T,1);      % computed beta value at each boosting step
alpha = zeros(T,1);     % computed alpha value at each boosting step
CLASSIFIER.rects = {}; CLASSIFIER.thresh = []; CLASSIFIER.cols = {}; CLASSIFIER.pol = []; CLASSIFIER.alpha = [];

for t = 1:T
    disp(' ');
    disp('===================================================================');
    disp(['              BOOSTING ROUND t = ' num2str(t) ',  ' EXP_NAME]);
    disp('-------------------------------------------------------------------');
    
    % randomly sample shape features
    get_random_shapes;

    % sample a subset of the whole training set to find optimal params
    sample_examples;
    
    % populate the feature responses for the sampled features
    disp('...optimizing sampled features.'); tic;
    [thresh p e ind] = optimize_features(Dsub,Wsub,Lsub,N_features,f_rects,f_cols,f_areas);
    to = toc; disp(['   Selected ' f_types{ind} ' feature ' num2str(ind) '. RANK = ' num2str(length(f_rects{ind})) ' thresh = ' num2str(thresh) '. Polarity = ' num2str(p) '. Time ' num2str(to) ' s.']);

    % visualize the selected feature
    rect_vis_ind(zeros(IMSIZE), f_rects{ind}, f_cols{ind}, p); 
    
    
    %% compute error. sanity check: best weak learner should beat 50%
    % +class < thresh < -class for pol=1. -class < thresh < +class for pol=-1
    E = AdaBoostClassifyDynamicA_mex(f_rects(ind), f_cols(ind), f_areas(ind),thresh, p, 1, D);
    %E = AdaBoostClassifyDynamicA_mex(f_rects(ind), f_cols(ind), f_areas(ind),thresh, 1, p, D);
    correct_classification = (E == L); incorrect_classification = (E ~= L);
    error(t) = sum(W .* incorrect_classification);
    disp(['...sampled weighted error = ' num2str(e) ' global weighted error = ' num2str(error(t))]);

    
    %% update the weights
    beta(t) = error(t) / (1 - error(t) );
    alpha(t) = log(1/beta(t));
    W = W .* ((beta(t)*ones(size(W))).^(1-incorrect_classification));
    W = normalize_weights(W);
    
    % add the new weak learner to the strong classifier
    CLASSIFIER.rects{t} = f_rects{ind}; 
    CLASSIFIER.cols{t} = f_cols{ind};
    CLASSIFIER.areas{t} = f_areas{ind};
    CLASSIFIER.pol(t) = p;
    CLASSIFIER.thresh(t) = thresh;
    CLASSIFIER.alpha(t) = alpha(t);
    CLASSIFIER.types{t} = f_types{ind};
    CLASSIFIER.norm = NORM;
    CLASSIFIER.method = RectMethod;
 
    
    % evaluate strong classifier performance, if desired (expensive)
    adaboost_eval;
   
    % store a temporary copy of boosted classifier
    save([results_folder EXP_NAME '-' host '-' date '.mat'], 'CLASSIFIER', 'W', 'stats', 'error');
    disp(['...saved as ' EXP_NAME '-' host '-' date '.mat']);
    
    % check for convergence (?)

end





%     % populate the feature responses for the sampled features
%     disp('...computing feature responses for the selected features.'); tic;
%     F = zeros(size(Dsub,1), size(f_rects,1), 'single');
%     for i = 1:N_features
%         F(:,i) = haar_featureDynamicA(Dsub, f_rects{i}, f_cols{i}, f_areas{i});
%     end; to = toc; disp(['   Elapsed time (MATLAB) ' num2str(to) ' seconds.']);
%     %tic; Fmex = HaarFeature_mex(Dsub, f_rects(:)', f_cols(:)');
%     %to=toc;  disp(['   Elapsed time (MEX) ' num2str(to) ' seconds.']);
%     
%     
%     %% find the best weak learner
%     disp('...selecting the best weak learner and its parameters.');
%     tic; [thresh p e ind] = best_weak_learner(Wsub,Lsub,F);     % subset of training data
%     %tic; [thresh p e ind] = best_weak_learner(W,L,F);          % entire set of training data
    