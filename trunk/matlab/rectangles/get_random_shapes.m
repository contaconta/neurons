% randomly sample features to use in this round of boosting

disp(['...generating ' num2str(N_features) ' Rank [2 to ' num2str(RANK) '] ' RectMethod ' rectangles.']);
f_rects = cell(N_features, 1); f_cols = cell(N_features,1); f_types = cell(N_features,1);
switch lower(RectMethod)
    case 'viola-jones'
        inds = randsample(size(pre_rects,1), N_features);
        f_rects = pre_rects(inds);  % randomly selected rectangles
        f_cols = pre_cols(inds);   % associated polarities
        f_types(1:N_features) = deal({'VJ'});
    case 'papageorgiou'
        inds = randsample(size(pre_rects,1), N_features);
        f_rects = pre_rects(inds);  % randomly selected rectangles
        f_cols = pre_cols(inds);   % associated polarities
        f_types(1:N_features) = deal({'PG'});
    case 'karim1'
        [tempr, tempc, f_rects, f_cols] = generate_rectangles(N_features, IMSIZE, RANK); 
        f_types(1:N_features) = deal({'Karim1'});
        clear tempr tempc;
    case 'simple'
        [f_rects, f_cols] = generate_simple_rectangles(N_features, IMSIZE, RANK);
        f_types(1:N_features) = deal({'Simple'});
    case 'rank-fixed'
        [f_rects, f_cols] = generate_simple_rectangles_rank_fixed(N_features, IMSIZE, RANK);
        f_types(1:N_features) = deal({'rank-fixed'});
    case 'lienhart'
        % sample pregenerated lienhart features
        inds = randsample(size(pre_rects,1), N_features);
        f_rects = pre_rects(inds);  % randomly selected rectangles
        f_cols = pre_cols(inds);   % associated polarities
        f_areas = pre_areas(inds);  % associated areas
        f_types = pre_types(inds);  % associated types
    case 'lienhartno3'
        % sample pregenerated lienhart features
        inds = randsample(size(pre_rects,1), N_features);
        f_rects = pre_rects(inds);  % randomly selected rectangles
        f_cols = pre_cols(inds);   % associated polarities
        f_areas = pre_areas(inds);  % associated areas
        f_types = pre_types(inds);  % associated types
    case 'kevin'
        [tempr, tempc, f_rects, f_cols] = generate_rectangles2(N_features, IMSIZE, RANK, CONNECTEDNESS);
        f_types(1:N_features) = deal({'Kevin'});
    case 'lisymm'
        [f_rects, f_cols] = generate_li_rectangles(N_features, IMSIZE, 1);
        f_types(1:N_features) = deal({'lisymm'});
    case 'liasymm'
        [f_rects, f_cols] = generate_li_rectangles(N_features, IMSIZE, 0);
        f_types(1:N_features) = deal({'liasymm'});
    case 'ramirez'
        [f_rects, f_cols] = generate_ramirez_rectangles(N_features, IMSIZE);
        f_types(1:N_features) = deal({'ramirez'});
    case 'asymmetric-mix'
        N1 = round( mixrate * N_features );
        N2 = N_features - N1;
        [f_rects(1:N1), f_cols(1:N1)] = generate_asymmetric_rectangles(N1, [24 24], 4, CONNECTEDNESS);
        f_types(1:N1) = deal({'Asymmetric'});
        inds = randsample(size(pre_rects,1), N2);
        f_rects(N1+1:N_features) = pre_rects(inds);  % randomly selected rectangles
        f_cols(N1+1:N_features) = pre_cols(inds);   % associated polarities
        f_types(N1+1:N_features) = deal({'VJ'});
        CLASSIFIER.mixrate = mixrate;
    case 'vjspecial'
        inds = randsample(size(pre_rects,1), N_features);
        f_rects = pre_rects(inds);  % randomly selected rectangles
        f_cols = pre_cols(inds);   % associated polarities
        f_types(1:N_features) = deal({'VJSPECIAL'});
    case 'mixed50'
        N1 = round(N_features/2);  N2 = N_features - N1;
        inds = randsample(size(pre_rects,1), N1);
        f_types(1:N1) = deal({'VJ'});
        f_types(N1+1:N_features) = deal({'Karim1'});
        f_rects(1:N1) = pre_rects(inds);  % randomly selected rectangles
        f_cols(1:N1) = pre_cols(inds);   % associated polarities
        [tempr, tempc, f_rects(N1+1:N_features), f_cols(N1+1:N_features)] = generate_rectangles(N2, IMSIZE, RANK);  
        clear tempr tempc N1 N2;
    case 'mixed33'
        N1 = round(N_features/3); N2 = round(N_features/3);  N3 = N_features - N1 - N2;
        inds = randsample(size(pre_rects,1), N1);
        f_rects(1:N1) = pre_rects(inds);  % randomly selected rectangles
        f_cols(1:N1) = pre_cols(inds);   % associated polarities
        f_types(1:N1) = deal({'VJ'});
        f_types(N1+1:N2) = deal({'Karim1'});
        f_types(N1+N2+1:N_features) = deal({'Simple'});
        [tempr, tempc, f_rects(N1+1:N1+N2), f_cols(N1+1:N1+N2)] = generate_rectangles(N2, IMSIZE, RANK); 
        [f_rects(N1+N2+1:N_features), f_cols(N1+N2+1:N_features)] = generate_simple_rectangles(N3, IMSIZE, RANK);
        clear tempr tempc N1 N2 N3;
    otherwise
        disp('Invalid Rectangle Generation Method was specified in settings! Stopping.');
        keyboard;
end


%% generate a list of [white black] areas. set to [0 0] if they are equal
%% or normalization is turned off
if ~exist('f_areas', 'var')
    f_areas = compute_areas2(IMSIZE, NORM, f_rects, f_cols);
end



%     %% TEMPORARY VISUALIZATION
%     figure(34334); disp('   VISUALIZING FEATURES');
%     for i = 1:N_features
%         rect_vis_ind(zeros(IMSIZE), f_rects{i}, f_cols{i});
%     end
%     
%    keyboard;