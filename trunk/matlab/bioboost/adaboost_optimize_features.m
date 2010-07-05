function [best_thresh best_pol best_err best_ind] = adaboost_optimize_features(D, W, L, N, f_inds)

thresh = zeros(N, 1);
pol = zeros(size(thresh));
err = zeros(size(thresh));

TPOS = sum( W(L==1));                  % Total sum of class 1 example weights
TNEG = sum( W(L==-1));                 % Total sum of class -1 example weights

% loop through the features
for i = 1:N
    
    % compute responses of feature i to the data set
    F = double(D(:, f_inds(i)));
       
    % get the optimal classification error, polarity, thresh for feature i (MEX CODE)
    [thresh(i), pol(i), err(i)] = adaboost_optimal_feature_params2(F, L, W, TPOS, TNEG); 
       
end

[best_err, best_ind] = min(err);
best_thresh = thresh(best_ind);
best_pol = pol(best_ind);


%keyboard;