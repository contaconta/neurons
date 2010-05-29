% randomly sample features to use in this round of boosting

disp(['...generating ' num2str(N_features) ' Rank [2 to ' num2str(RANK) '] ' RectMethod ' rectangles.']);
switch RectMethod
    case 'Viola-Jones'
        inds = randsample(size(N,1), N_features);
        f_rects = N(inds);  % randomly selected rectangles
        f_cols = P(inds);   % associated polarities
    case 'Karim1'
        [tempr, tempc, f_rects, f_cols] = generate_rectangles(N_features, IMSIZE, RANK);  
        clear tempr tempc;
    case 'Simple'
        [f_rects, f_cols] = generate_simple_rectangles(N_features, IMSIZE, RANK);
    case 'Kevin'
        [tempr, tempc, f_rects, f_cols] = generate_rectangles2(N_features, IMSIZE, RANK, CONNECTEDNESS);
end
if ANORM; f_areas = compute_areas(IMSIZE, f_rects); end;  



%     %%% TEMPORARY VISUALIZATION
%     figure(34334); disp('   VISUALIZING FEATURES');
%     for i = 1:N_features
%         rect_vis_ind(zeros(IMSIZE), f_rects{i}, f_cols{i});
%     end