function [best_thresh, best_pol, min_err, best_ind] = best_weak_learner(W, L, F)


% determine the optimal threshold and polarity for each feature
thresh = zeros(size(F,2), 1);
pol = zeros(size(thresh));
err = zeros(size(thresh));
thresh2 = zeros(size(thresh));
pol2 = zeros(size(thresh));
err2 = zeros(size(thresh));

tic;
for f = 1:size(F,2)    
    [thresh2(f), pol2(f), err2(f)] = optimal_feature_params2(F(:,f), L, W);
end
toc;

tic; 
for f = 1:size(F,2)    
    [thresh(f), pol(f), err(f)] = optimal_feature_params(F(:,f), L, W); 
end
toc;


% find the minimum error, and optimum threshold, polarity
[min_err, best_ind] = min(err);
best_thresh = thresh(best_ind);
best_pol = pol(best_ind);

%plot_N_learners(CLASSIFIER, N, IMSIZE)

keyboard;