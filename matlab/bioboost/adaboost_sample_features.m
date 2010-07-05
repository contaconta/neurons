

N_features = min(N_features, size(D,2));

f_inds = randsample(size(D,2), N_features, true);
f_inds = f_inds';