% randomly sample features to use in this round of boosting

disp(['...generating ' num2str(N_features) ' Rank [2 to ' num2str(RANK) '] ' RectMethod ' rectangles.']);
f_rects = cell(N_features, 1); f_cols = cell(N_features,1); f_types = cell(N_features,1);
switch RectMethod
    case 'Viola-Jones'
        inds = randsample(size(N,1), N_features);
        f_rects = N(inds);  % randomly selected rectangles
        f_cols = P(inds);   % associated polarities
        f_types(1:N_features) = deal({'VJ'});
    case 'Karim1'
        [tempr, tempc, f_rects, f_cols] = generate_rectangles(N_features, IMSIZE, RANK); 
        f_types(1:N_features) = deal({'Karim1'});
        clear tempr tempc;
    case 'Simple'
        [f_rects, f_cols] = generate_simple_rectangles(N_features, IMSIZE, RANK);
        f_types(1:N_features) = deal({'Simple'});
    case 'Kevin'
        [tempr, tempc, f_rects, f_cols] = generate_rectangles2(N_features, IMSIZE, RANK, CONNECTEDNESS);
        f_types(1:N_features) = deal({'Kevin'});
    case 'VJSPECIAL'
        inds = randsample(size(N,1), N_features);
        f_rects = N(inds);  % randomly selected rectangles
        f_cols = P(inds);   % associated polarities
        f_types(1:N_features) = deal({'VJSPECIAL'});
    case 'Mixed50'
        N1 = round(N_features/2);  N2 = N_features - N1;
        inds = randsample(size(N,1), N1);
        f_types(1:N1) = deal({'VJ'});
        f_types(N1+1:N_features) = deal({'Karim1'});
        f_rects(1:N1) = N(inds);  % randomly selected rectangles
        f_cols(1:N1) = P(inds);   % associated polarities
        [tempr, tempc, f_rects(N1+1:N_features), f_cols(N1+1:N_features)] = generate_rectangles(N2, IMSIZE, RANK);  
        clear tempr tempc N1 N2;
    case 'Mixed33'
        N1 = round(N_features/3); N2 = round(N_features/3);  N3 = N_features - N1 - N2;
        inds = randsample(size(N,1), N1);
        f_rects(1:N1) = N(inds);  % randomly selected rectangles
        f_cols(1:N1) = P(inds);   % associated polarities
        f_types(1:N1) = deal({'VJ'});
        f_types(N1+1:N2) = deal({'Karim1'});
        f_types(N1+N2+1:N_features) = deal({'Simple'});
        [tempr, tempc, f_rects(N1+1:N1+N2), f_cols(N1+1:N1+N2)] = generate_rectangles(N2, IMSIZE, RANK); 
        [f_rects(N1+N2+1:N_features), f_cols(N1+N2+1:N_features)] = generate_simple_rectangles(N3, IMSIZE, RANK);
        clear tempr tempc N1 N2 N3;
    otherwise
        error('invalid Rectangle Generation Method was specified in settings.');
end


%% generate a list of [white black] areas. set to [0 0] if they are equal
%% or normalization is turned off
f_areas = compute_areas2(IMSIZE, NORM, f_rects, f_cols);


% if ANORM; 
%     f_areas = compute_areas2(IMSIZE, f_rects, f_cols); 
% else
%     f_areas = compute_nonorm_areas(f_rects);
% end


%     %%% TEMPORARY VISUALIZATION
%     figure(34334); disp('   VISUALIZING FEATURES');
%     for i = 1:N_features
%         rect_vis_ind(zeros(IMSIZE), f_rects{i}, f_cols{i});
%     end
%     
%     keyboard;