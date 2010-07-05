function [best_thresh, best_pol, best_err] = adaboost_optimal_feature_params2(F1, L, W, TPOS, TNEG)

[Fs inds] = sort(F1);   % sorted feature responses
Ls = L(inds);           % sorted class labels
Ws = W(inds);           % sorted weights

[err, pol] = ChooseThreshold_mex(Fs, Ws, Ls, TPOS, TNEG);

[best_err, q_ind]   = min(err);
best_thresh         = Fs(q_ind);
best_pol        	= pol(q_ind);

% !!! THE THRESHOLD SHOULD BE BETWEEN q_ind and q_ind-1


if (best_err > 1) || (best_err < 0)
    plot(err);
    hold on;
    plot(pol, 'r-');
    keyboard;
end

%keyboard;