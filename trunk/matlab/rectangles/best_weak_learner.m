function [best_thresh, best_pol, min_err, best_ind] = best_weak_learner(W, L, F)


% determine the optimal threshold and polarity for each feature
thresh = zeros(size(F,2), 1);
pol = zeros(size(thresh));
err = zeros(size(thresh));
for f = 1:size(F,2)    
    [thresh(f), pol(f), err(f)] = optimal_feature_params(F(:,f), L, W);    
end


% find the minimum error, and optimum threshold, polarity
[min_err, best_ind] = min(err);
best_thresh = thresh(best_ind);
best_pol = pol(best_ind);

%plot_N_learners(CLASSIFIER, N, IMSIZE)
