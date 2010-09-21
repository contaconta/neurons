

N_features = min(N_features, size(D,2));

%f_inds = randsample(size(D,2), N_features, true);
%f_inds = f_inds';


% do not sample the features here!
%disp(['   ' num2str(N_features) ' features used ']);
f_inds = 1:N_features;