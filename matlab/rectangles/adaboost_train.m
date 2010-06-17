%% LOAD PARAMETERS
adaboost_settings;


%% PRE-BOOSTING
tstart = tic;

% pre-generate necessary feature pools (VJ, Lienhart, ...)
adaboost_pregenerate_features;

% load the database into D
adaboost_load_database;

% initialize the weights, set each class to have equal weights initially
W = ones(size(L));          % example weights
W(L == 1) = .5 * (W(L == 1) / sum(W(L == 1)));
W(L == -1) = .5 * (W(L == -1) / sum(W(L == -1)));


%% PERFORM BOOSTING

[pathstr, name, ext, versn] = fileparts([results_folder EXP_NAME '-' host '-' date '.mat']);
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
   	clear f_areas f_rects f_cols f_types;
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
 
    
    
    % TEMP AREA SANITY CHECK CODE
    %[r c] = ind2sub([25 25], CLASSIFIER.rects{t}{1});
    a = compute_areas2([24 24], NORM, CLASSIFIER.rects(t), CLASSIFIER.cols(t));
    disp([  '   CLASSIFIER.areas=[' num2str(CLASSIFIER.areas{t})  '] a=[' num2str(a{1}) ']']);
    
    
    
    % evaluate strong classifier performance, if desired (expensive)
    adaboost_eval;
   
    % store a temporary copy of boosted classifier
    save([pathstr '/' name ext], 'CLASSIFIER', 'W', 'stats', 'error'); disp(['...saved as ' name ext]);

end


%% evaluate the results on the test set!
adaboost_self_evaluate([pathstr '/' name ext]);

tstop = toc(tstart);
cmd = ['mv ' pathstr '/' name ext ' ' pathstr '/finished/' name ext];
system(cmd); 
disp('==================================================================');
disp(['  ' name ' finished processing in ' num2str(round(tstop)) ' seconds. Moved to /finished/']);
disp('==================================================================');
%send_gmail({'kevin.smith@epfl.ch'}, ['finished ' name], [host ' has completed ' name ' in ' num2str(round(tstop)) ' seconds.']);



