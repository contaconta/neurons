% randomly sample features for this round of boosting
if VJ == 1
    disp(['...sampling ' num2str(N_features) ' Viola-Jones rectangles.']);        
    inds = randsample(size(N,1), N_features);
    f_rects = N(inds);  % randomly selected rectangles
    f_cols = P(inds);   % associated polarities
else
    disp(['...generating ' num2str(N_features) ' Rank [2 to ' num2str(RANK) '] rectangles.']);
    %[tempr, tempc, f_rects, f_cols] = generate_rectangles(N_features, IMSIZE, RANK);
    %[tempr, tempc, f_rects, f_cols] = generate_rectangles2(N_features, IMSIZE, RANK, CONNECTEDNESS);
    [f_rects, f_cols] = generate_simple_rectangles(N_features, IMSIZE, RANK);
    clear tempr tempc;
end
if ANORM; f_areas = compute_areas(IMSIZE, f_rects); end;  

%     %%% TEMPORARY VISUALIZATION
%     figure(34334); disp('   VISUALIZING FEATURES');
%     for i = 1:N_features
%         rect_vis_ind(zeros(IMSIZE), f_rects{i}, f_cols{i});
%     end