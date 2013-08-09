function evaluateCellTracking(seq_num)


seq         = sprintf('%03d', seq_num);  %'001';
est_folder  = '/home/ksmith/data/sinergia_evaluation/Detections10x/';
gt_folder   = '/home/ksmith/data/sinergia_evaluation/annotations/somatool/';
C           = 696;
R           = 520;
Tmax        = 97;

% load the GT and EST files
filename = sprintf('%s%s.mat', est_folder,seq);
load(filename);
filename = sprintf('%s%s.mat', gt_folder,seq);
load(filename);



% convert the ground truth to structure
GT = convert_GT_to_S(GT); %#ok<NODEF>

% convert the est to structure
EST = convert_EST_to_S(Sequence);


img_folder = sprintf('%s%s/green/','/home/ksmith/data/sinergia_evaluation/Selection10x/',seq);
mv = loadMovie(img_folder);







coverage_threshold = .33;
FPcount = zeros(Tmax,1);
FNcount = zeros(Tmax,1);
MTcount = zeros(Tmax,1);
MOcount = zeros(Tmax,1);
FIcount = zeros(Tmax,1);
OKGTcount = zeros(Tmax,1);
OKESTcount = zeros(Tmax,1);
GTperT  = zeros(Tmax,1);
accumulated_coverage = zeros(numel(EST), numel(GT));

for t = 1:Tmax
    fprintf('evaluating configuration t=%d/%d\n', t, Tmax);
    % make the configuration map
    coverage_raw = zeros(numel(EST), numel(GT));
    for i = 1:numel(EST)
        if ~isempty(EST(i).P(t).x)
            for j = 1:numel(GT)
                if ~isempty(GT(j).P(t).x)
                    coverage_raw(i,j) = coverage_test(EST(i).P(t), GT(j).P(t));
                end
            end
        end
    end
    coverage{t} = coverage_raw >= coverage_threshold; %#ok<AGROW>
    accumulated_coverage = accumulated_coverage + coverage{t};

    % GT CONFIGURATION
    for i = 1:numel(GT)
        if ~isempty(GT(i).P(t).x)
            conf_vec = coverage{t}(:,i);
            num_est_covered = sum(conf_vec);
            GTperT(t) = GTperT(t) + 1;
            switch num_est_covered
                case 0
                    GT(i).P(t).status = 'FN';  FNcount(t) = FNcount(t) + 1;
                case 1
                    GT(i).P(t).status = 'OK';  OKGTcount(t) = OKGTcount(t) + 1;
                otherwise
                    GT(i).P(t).status = 'MT';  MTcount(t) = MTcount(t) + 1;
            end           
        else
            GT(i).P(t).status = '-';
        end
    end
    
    % EST CONFIGURATION
    for i = 1:numel(EST)
        if ~isempty(EST(i).P(t).x)
            conf_vec = coverage{t}(i,:);
            num_gt_covered = sum(conf_vec);
            switch num_gt_covered
                case 0
                    EST(i).P(t).status = 'FP';  FPcount(t) = FPcount(t) + 1;
                case 1
                    EST(i).P(t).status = 'OK';  OKESTcount(t) = OKESTcount(t) + 1;
                otherwise
                    EST(i).P(t).status = 'MO';  MOcount(t) = MOcount(t) +1;
            end
        else
            EST(i).P(t).status = '-';
        end
    end

%     figure(1);
%     imagesc(accumulated_coverage);
%     xlabel('GT id');
%     ylabel('EST id');
%     drawnow;
%     pause(0.1);
end



fprintf('evaluating identification\n');
id_map_GT_tracked_by = zeros(numel(GT),1);
for i = 1:numel(GT)
    vec = accumulated_coverage(:,i);
    [val, ind] = max(vec);     %#ok<ASGLU>
    id_map_GT_tracked_by(i) = ind;
end
  

id_map_EST_tracks = zeros(numel(EST),1);
for i = 1:numel(EST)
    vec = accumulated_coverage(i,:);
    [val, ind] = max(vec);     %#ok<ASGLU>
    id_map_EST_tracks(i) = ind;
end



for t = 1:Tmax
    for i = 1:numel(GT)
        vec = coverage{t}(:,i);
        inds = find(vec);
        for k = 1:numel(inds)
            ind = inds(k);
            if ind == id_map_GT_tracked_by(i)
                % everything is okay
            else
                % not the tracker we want
                switch EST(ind).P(t).status
                    case 'OK'
                        EST(ind).P(t).status = 'FI';
                        FIcount(t) = FIcount(t) + 1;
                    otherwise
                        % do nothing
                end
            end
        end
    end
    
    
    for i = 1:numel(EST)
        vec = coverage{t}(i,:);
        inds = find(vec);
        for k = 1:numel(inds)
            ind = inds(k);
            if ind == id_map_EST_tracks(i)
                % everything is okay
            else
                % not the tracker we want
                switch EST(i).P(t).status
                    case 'OK'
                        EST(i).P(t).status = 'FI';
                        FIcount(t) = FIcount(t) + 1;
                    otherwise
                        % do nothing
                end
            end
        end

    end
    
end







%% render the result movie
figure('Position', [1937 262 696 520]); hold on; 
axis([0 C 0 R]);
imagesc(mv{1});
newmv = {};
colorFN = [1 .498 .3137];
colorFP = [1 .2 .2];
colorMT = [1 .2 1];
colorMO = [.7216 .4510 .2];
colorFI = [0 1 0];
colorOKest = [.3 .3 1];
colorOKgt  = [.5 .5 .5];

for t = 1:Tmax
    cla; gtp = []; estp = [];
    set(gca, 'Position', [0 0 1 1]);
    
    % show the image
    imagesc(mv{t});
    
    
    % render each ground truth
    for i = 1:numel(GT)
        switch GT(i).P(t).status
            case 'FN'
                gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', colorFN, 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
                str = sprintf('%dgt FN',i);
                text(max(GT(i).P(t).x)+5, min(GT(i).P(t).y(1)), str, 'Color', colorFN);
            case 'OK'
                gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', colorOKgt, 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
                str = sprintf('%dgt',i);
                text(max(GT(i).P(t).x)+5, min(GT(i).P(t).y(1)), str);
            case 'MT'
                gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', colorMT, 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
                str = sprintf('%dgt MT',i);
                text(max(GT(i).P(t).x)+5, min(GT(i).P(t).y(1)), str, 'Color', colorMT);
            case '-'
                % nothing to render
        end
    end
    
    
     % render each estimate
     for i = 1:numel(EST)
            switch EST(i).P(t).status
                case 'FP'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', colorFP, 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest FP',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y-5), str,'Color', colorFP);
                case 'OK'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', [0 0 1], 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y)-5, str, 'Color', colorOKest);
                case 'MO'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', colorMO, 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest MO',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y-5), str,'Color', colorMO);
                case 'FI'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', colorFI, 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest FI',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y-5), str,'Color', colorFI);
                case '-'
                    % nothing to render
            end
     end
    
    
	% display the current error count
    patch([5 110 110 5], [5 5 70 70], 2, 'FaceColor', [1 1 1], 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
    text(10, 10, 2, sprintf('%04d', sum(OKGTcount(1:t))));
    text(10, 60, 2,sprintf('%04d', sum(FPcount(1:t))));
    text(10, 50, 2,sprintf('%04d', sum(FNcount(1:t))));
    text(10, 40, 2,sprintf('%04d', sum(MTcount(1:t))));
    text(10, 30, 2,sprintf('%04d', sum(MOcount(1:t))));
    text(10, 20, 2,sprintf('%04d', sum(FIcount(1:t))));
    text(50, 10, 2,'Correct', 'Color', colorOKest);
    text(50, 60, 2,'FP','Color', colorFP);
    text(50, 50, 2,'FN','Color', colorFN);
    text(50, 40, 2,'MT','Color', colorMT);
    text(50, 30, 2,'MO','Color', colorMO);
    text(50, 20, 2,'FI','Color', colorFI);
    
    
    pause(.1);
    refresh;
    drawnow;
    pause(0.1);

    
    set(gcf, 'Position', [1937 262 696 520]);
    F = getframe(gcf);
    I = F.cdata;
    newmv{t} = I; %#ok<AGROW>
     
end



filename = sprintf('%sannotation%s.mat', '/home/ksmith/data/sinergia_evaluation/annotations/celltracking/', seq);
fprintf('saving %s\n', filename);
save(filename, 'GT', 'EST', 'FNcount', 'FPcount', 'MTcount', 'MOcount', 'FIcount', 'OKGTcount', 'OKESTcount');


error_rate = sum(FNcount + FPcount + MTcount + MOcount + FIcount) / sum(FNcount + FPcount + MTcount + MOcount + OKGTcount + OKESTcount + FIcount);
fprintf('error_rate = %1.3f   %d FN  %d FP  %d MT  %d MO  %d FI  %d OK\n', error_rate, sum(FNcount), sum(FPcount), sum(MTcount), sum(MOcount), sum(FIcount), sum(OKGTcount+OKESTcount));




outputFolder = '/home/ksmith/data/sinergia_evaluation/output/';
movfile = sprintf('eval%03d', seq_num);
trkMovie(newmv, outputFolder, outputFolder, movfile); fprintf('\n');
fprintf('wrote to %s%s.mp4\n', outputFolder,movfile);


