function [best_thresh best_pol best_err best_ind best_w] = optimize_features_separea(D, W, L, N, rects, brute_lists, separeas)

thresh = zeros(N, 1);
pol = zeros(size(thresh));
err = zeros(size(thresh));

w = cell(size(thresh));

TPOS = sum( W(L==1));                  % Total sum of class 1 example weights
TNEG = sum( W(L==-1));                 % Total sum of class -1 example weights


tic;

% loop through the features
for i = 1:N
    
    
    mus = get_mus(rects{i}, D, separeas{i});

    
    RANK = length(rects{i});
    b_list = brute_lists{RANK};
    err_list = zeros(size(b_list,1),1); thresh_list = err_list; pol_list = err_list;
    
    for j = 1:size(b_list,1)
        weight = b_list(j,:);
        F = mus*weight';
        [thresh_list(j), pol_list(j), err_list(j)] = optimal_feature_params2(F, L, W, TPOS, TNEG); 
    end
    
    
 
    [v bestind] = min(err_list);
    thresh(i) = thresh_list(bestind);
    pol(i) = pol_list(bestind);
    err(i) = err_list(bestind);
    
    w{i} = b_list(bestind,:);
    
end

[best_err, best_ind] = min(err);
best_thresh = thresh(best_ind);
best_pol = pol(best_ind);
best_w = w{best_ind};


to = toc;
disp(['FUCKING OPTIMAL WEIGHTS ARE [' num2str(best_w) '] time = ' num2str(to) ]);



function mus = get_mus(rects, D, separeas)

%mus = cell(size(rects));
mus = zeros(size(D,1), length(rects));



for i = 1:length(rects)
    
    rect = rects{i};
    
    %[r1 c1] = ind2sub([25 25], rect);
    %a1 = (r1(4)-r1(1))  * (c1(4)-c1(1));
    SUM_rect = (D(:,rect(1)) + D(:,rect(4)) - D(:,rect(3)) - D(:,rect(2)));
    mus(:,i) = SUM_rect/separeas(i);
end



% function [u1 u2] = u1u2(rects,cols,D)
% 
%   
% 
% if cols(1) == 1
%     rect1 = rects{1};
%     rect2 = rects{2};
%     [r1 c1] = ind2sub([25 25], rect1);
%     [r2 c2] = ind2sub([25 25], rect2);
%     a1 = (r1(4)-r1(1))  * (c1(4)-c1(1));
%     a2 = (r2(4)-r2(1))  * (c2(4)-c2(1));
% else
%     rect1 = rects{2};
%     rect2 = rects{1};
%     [r1 c1] = ind2sub([25 25], rect1);
%     [r2 c2] = ind2sub([25 25], rect2);
%     a1 = (r1(4)-r1(1))  * (c1(4)-c1(1));
%     a2 = (r2(4)-r2(1))  * (c2(4)-c2(1));
% end
% u1 = (D(:,rect1(1)) + D(:,rect1(4)) - D(:,rect1(3)) - D(:,rect1(2)))/a1;
% u2 = (D(:,rect2(1)) + D(:,rect2(4)) - D(:,rect2(3)) - D(:,rect2(2)))/a2;

