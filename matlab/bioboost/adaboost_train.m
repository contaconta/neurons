%% LOAD PARAMETERS
adaboost_settings;


%% PRE-BOOSTING
tstart = tic;

% pre-generate necessary feature pools 
% not used

% load the database into D
adaboost_load_database;

% initialize the weights, set each class to have equal weights initially
W = ones(size(L));          % example weights
W = W/sum(W);
%W(L == 1) = .5 * (W(L == 1) / sum(W(L == 1)));
%W(L == -1) = .5 * (W(L == -1) / sum(W(L == -1)));


%% PERFORM BOOSTING

[pathstr, name, ext, versn] = fileparts([results_folder EXP_NAME '-' host '-' date '.mat']);
stats = zeros(T,7);     % keep statistics for each boosting round
error = zeros(T,1);     % weighted classification error at each boosting step
beta = zeros(T,1);      % computed beta value at each boosting step
alpha = zeros(T,1);     % computed alpha value at each boosting step
CLASSIFIER.inds = {}; CLASSIFIER.thresh = []; CLASSIFIER.pol = []; CLASSIFIER.alpha = [];

for t = 1:T
    disp(' ');
    disp('===================================================================');
    disp(['              BOOSTING ROUND t = ' num2str(t) ',  ' EXP_NAME]);
    disp('-------------------------------------------------------------------');
    
    % randomly sample features
   	adaboost_sample_features

    % sample a subset of the whole training set to find optimal params
    adaboost_sample_examples;
    
    % populate the feature responses for the sampled features
    disp('...optimizing sampled features.'); tic;
    [thresh p e ind] = adaboost_optimize_features(Dsub, Wsub, Lsub, N_features, f_inds);
   
    to = toc; disp(['   Selected feature ' num2str(ind) '.  thresh = ' num2str(thresh) '. Polarity = ' num2str(p) '. Time ' num2str(to) ' s.']);

    %figure(1);  bar(W(1:3000)); drawnow; pause(0.01);
    
    %% compute error. sanity check: best weak learner should beat 50%
    % +class < thresh < -class for pol=1. -class < thresh < +class for pol=-1
    
%     F = D(:, f_inds(ind));
%     E =  double((p*F < p*thresh));
%     E(E == 0) = -1;
    %E = AdaBoostClassifyDynamicA_mex(f_rects(ind), f_cols(ind), f_areas(ind),thresh, p, 1, D);
  	E = AdaBoostClassify_mex({f_inds(ind)}, thresh, p, 1, D);
   
    correct_classification = (E == L); incorrect_classification = (E ~= L);
    error(t) = sum(W .* incorrect_classification);
    disp(['...sampled weighted error = ' num2str(e) ' global weighted error = ' num2str(error(t))]);

 	%keyboard;
    
    %% update the weights
    beta(t) = error(t) / (1 - error(t) );
    alpha(t) = log(1/beta(t));
    W = W .* ((beta(t)*ones(size(W))).^(1-incorrect_classification));
    W = adaboost_normalize_weights(W);
    
    % add the new weak learner to the strong classifier
    CLASSIFIER.inds{t} = f_inds(ind); 
    %CLASSIFIER.cols{t} = f_cols{ind};
    %CLASSIFIER.areas{t} = f_areas{ind};
    CLASSIFIER.pol(t) = p;
    CLASSIFIER.thresh(t) = thresh;
    CLASSIFIER.alpha(t) = alpha(t);
    %CLASSIFIER.types{t} = f_types{ind};
    %CLASSIFIER.norm = NORM;
    %CLASSIFIER.method = RectMethod;
    CLASSIFIER.dataset = DATASET;
    
    
    % TEMP AREA SANITY CHECK CODE
    %[r c] = ind2sub([25 25], CLASSIFIER.rects{t}{1});
    %a = compute_areas2([24 24], NORM, CLASSIFIER.rects(t), CLASSIFIER.cols(t));
    %disp([  '   CLASSIFIER.areas=[' num2str(CLASSIFIER.areas{t})  '] a=[' num2str(a{1}) ']']);
    
    
    
    % evaluate strong classifier performance, if desired (expensive)
    adaboost_eval;
   
    % store a temporary copy of boosted classifier
    save([pathstr '/' name ext], 'CLASSIFIER', 'W', 'stats', 'error'); disp(['...saved as ' name ext]);

   
end


%% evaluate the results on the test set!
%adaboost_self_evaluate([pathstr '/' name ext], EVALUATE_LIST);

tstop = toc(tstart);
%cmd = ['mv ' pathstr '/' name ext ' ' pathstr '/finished/' name ext];
%system(cmd); 
disp('==================================================================');
disp(['  ' name ' finished processing in ' num2str(round(tstop)) ' seconds.']);
disp('==================================================================');
%send_gmail({'kevin.smith@epfl.ch'}, ['finished ' name], [host ' has completed ' name ' in ' num2str(round(tstop)) ' seconds.']);

disp(['saved classifier as ' name '.txt']);
classifier2text(CLASSIFIER, [name '.txt']);
