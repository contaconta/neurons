function [best_thresh best_pol best_err best_ind] = optimize_features(D, W, L, N, rects, cols, areas)

thresh = zeros(N, 1);
pol = zeros(size(thresh));
err = zeros(size(thresh));

TPOS = sum( W(L==1));                  % Total sum of class 1 example weights
TNEG = sum( W(L==-1));                 % Total sum of class -1 example weights

% loop through the features
for i = 1:N
    
    % compute responses of feature i to the data set
    F = haar_featureDynamicA(D, rects{i}, cols{i}, areas{i});
    %keyboard;
        
    % get the optimal classification error, polarity, thresh for feature i
    % MEX CODE
    [thresh(i), pol(i), err(i)] = optimal_feature_params2(F, L, W, TPOS, TNEG); 
    
    % SLOWER (MORE RELIABLE ?) 
    % MATLAB CODE
    %[thresh(i), pol(i), err(i)] = optimal_feature_params(F, L, W);    
end

[best_err, best_ind] = min(err);
best_thresh = thresh(best_ind);
best_pol = pol(best_ind);


