% 
% seq         = '001';
% est_folder  = '/home/ksmith/data/sinergia_evaluation/Detections10x/';
% gt_folder   = '/home/ksmith/data/sinergia_evaluation/annotations/';
% C           = 696;
% R           = 520;
% Tmax        = 97;
% 
% % load the GT and EST files
% filename = sprintf('%s%s.mat', est_folder,seq);
% load(filename);
% filename = sprintf('%s%s.mat', gt_folder,seq);
% load(filename);
% 
% 
% 
% % convert the ground truth to structure
% GT = convert_GT_to_S(GT);
% 
% % convert the est to structure
% EST = convert_EST_to_S(Sequence);
% 
% 
% img_folder = sprintf('%s%s/green/','/home/ksmith/data/sinergia_evaluation/Selection10x/',seq);
% mv = loadMovie(img_folder);
% 


figure('Position', [1926 446 1044 780+135]); hold on; 
% set(gcf, 'Position', [1926 446 round(696*1.5) round(520*1.5)]);
axis([0 C 0 R]);
imagesc(mv{1});
% pos = get(gcf, 'Position');
% set(gcf, 'Position', [pos(1) pos(2) round(1.5 * [696 520])]);
% keyboard;


coverage_threshold = .33;
FPcount = 0;
FNcount = 0;
MTcount = 0;
MOcount = 0;
OKGTcount = 0;
OKESTcount = 0;
GTperT  = zeros(Tmax,1);
newmv = {};

for t = 1:Tmax
    cla; gtp = []; estp = [];
    set(gca, 'Position', [0 0 1 1]);
    
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
    coverage = coverage_raw >= coverage_threshold;
    
    
    % show the image
    imagesc(mv{t});
    
    
    
    % determine status of each ground truth
    for i = 1:numel(GT)
        if ~isempty(GT(i).P(t).x)
            conf_vec = coverage(:,i);
            num_est_covered = sum(conf_vec);
            GTperT(t) = GTperT(t) + 1;
            switch num_est_covered
                case 0
                    GT(i).P(t).status = 'FN';  FNcount = FNcount + 1;
                case 1
                    GT(i).P(t).status = 'OK';  OKGTcount = OKGTcount + 1;
                otherwise
                    GT(i).P(t).status = 'MT';  MTcount = MTcount + 1;
            end
            
            % render
            switch GT(i).P(t).status
                case 'FN'
                    gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', [1 0 0], 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
                    str = sprintf('%dgt FN',i);
                    text(max(GT(i).P(t).x)+5, min(GT(i).P(t).y(1)), str, 'Color', [1 0 0]);
                case 'OK'
                    gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', [.5 .5 .5], 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
                    str = sprintf('%dgt',i);
                    text(max(GT(i).P(t).x)+5, min(GT(i).P(t).y(1)), str);
                case 'MT'
                    gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', [1 0 0], 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
                    str = sprintf('%dgt MT',i);
                    text(max(GT(i).P(t).x)+5, min(GT(i).P(t).y(1)), str, 'Color', [1 0 0]);
            end
        else
            GT(i).P(t).status = [];
        end
    end
    
    for i = 1:numel(EST)
        if ~isempty(EST(i).P(t).x)
            conf_vec = coverage(i,:);
            num_gt_covered = sum(conf_vec);
            switch num_gt_covered
                case 0
                    EST(i).P(t).status = 'FP';  FPcount = FPcount + 1;
                case 1
                    EST(i).P(t).status = 'OK';  OKESTcount = OKESTcount + 1;
                otherwise
                    EST(i).P(t).status = 'MO';  MOcount = MOcount +1;
            end
            
            
            % render
            switch EST(i).P(t).status
                case 'FP'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', [1 0 1], 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest FP',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y-5), str,'Color', [1 0 1]);
                case 'OK'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', [0 0 1], 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y)-5, str, 'Color', [.3 .3 1]);
                case 'MO'
                    estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', [1 0 1], 'FaceAlpha', .5, 'EdgeColor', 'none');
                    str = sprintf('%dest MO',i);
                    text(max(EST(i).P(t).x)+5, max(EST(i).P(t).y-5), str,'Color', [1 0 1]);
            end
            
            
        else
            EST(i).P(t).status = [];
        end
    end
    
    
    % display the current error count
    text(10, 10, sprintf('%04d  Correct', OKGTcount));
    text(10, 20, sprintf('%04d  FP', FPcount));
    text(10, 30, sprintf('%04d  FN', FNcount));
    text(10, 40, sprintf('%04d  MT', MTcount));
    text(10, 50, sprintf('%04d  MO', MOcount));
    
    
    
    
    pause(.1);
    refresh;
    drawnow;
    pause(0.1);

    
    set(gcf, 'Position', [1937 262 696 520]);
    F = getframe(gcf);
    I = F.cdata;
    newmv{t} = I;
end


% GTerror_rate = (FNcount + MTcount) /  (FNcount + MTcount + OKGTcount);
% ESTerror_rate = (FPcount + MOcount) /  (FPcount + MTcount + OKESTcount);
error_rate = (FNcount + FPcount + MTcount + MOcount) / (FNcount + FPcount + MTcount + MOcount + OKGTcount + OKESTcount);


% fprintf('GT_error_rate = %1.3f   %d FN  %d MT  %d OK\n', GTerror_rate, FNcount, MTcount, OKGTcount);
% fprintf('EST_error_rate = %1.3f   %d FN  %d MT  %d OK\n', ESTerror_rate, FPcount, MOcount, OKESTcount);

fprintf('error_rate = %1.3f   %d FN  %d FP  %d MT  %d MO  %d OK\n', error_rate, FNcount, FPcount, MTcount, MOcount, OKGTcount+OKESTcount);




outputFolder = '/home/ksmith/data/sinergia_evaluation/output/';
movfile = sprintf('eval%03d', seq);
trkMovie(newmv, outputFolder, outputFolder, movfile); fprintf('\n');
fprintf('wrote to %s%s.mp4\n', outputFolder,movfile);



% maxGT = max(GTperT);
% OKbar = sum(OKcount) / (Tmax * maxGT);
% FPbar = sum(FPcount) / (Tmax * maxGT);
% FNbar = sum(FNcount) / (Tmax * maxGT);
% MTbar = sum(MTcount) / (Tmax * maxGT);
% MObar = sum(MOcount) / (Tmax * maxGT);
% 
% 
% fprintf('OKbar = %1.2f\n', OKbar);
% fprintf('FPbar = %1.2f\n', FPbar);
% fprintf('FNbar = %1.2f\n', FNbar);
% fprintf('MTbar = %1.2f\n', MTbar);
% fprintf('MObar = %1.2f\n', MObar);





% figure; hold on;
% axis([0 C 0 R]);
%
% for t = 1:Tmax
%     cla; gtp = []; estp = [];
% %     axis off
%     imagesc(mv{t});
%     set(gca, 'Position', [0 0 1 1]);
%     for i = 1:numel(GT)
%         if ~isempty(GT(i).P(t).x)
%             gtp(i) = patch(GT(i).P(t).x, GT(i).P(t).y, 1, 'FaceColor', [.5 .5 .5], 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
%             str = sprintf('gt%d',i);
%             text(GT(i).P(t).x(1), GT(i).P(t).y(1), str);
%         end
%     end
%
%     for i = 1:numel(EST)
%         if ~isempty(EST(i).P(t).x)
%             estp(i) = patch(EST(i).P(t).x, EST(i).P(t).y, 1, 'FaceColor', [0 0 1], 'FaceAlpha', .25, 'EdgeColor', 'none');
%             str = sprintf('est%d',i);
%             text(EST(i).P(t).x(1), EST(i).P(t).y(1), str);
%         end
%     end
%
%
%     coverage = zeros(numel(EST), numel(GT));
%     for i = 1:numel(EST)
%         if ~isempty(EST(i).P(t).x)
%             for j = 1:numel(GT)
%                 if ~isempty(GT(j).P(t).x)
%                     coverage(i,j) = coverage_test(EST(i).P(t), GT(j).P(t));
%                 end
%             end
%         end
%     end
%
%
%     keyboard;
% end




% % clockwise vertices, start point = end point
% i = 1;
% t = 1;
% cmin = GT{i}(t,1);
% rmin = GT{i}(t,2);
% width = GT{i}(t,3);
% height = GT{i}(t,4);
%
% x1 = [cmin cmin        cmin+width  cmin+width cmin];
% y1 = [rmin rmin+height rmin+height rmin       rmin];
%
%
% i = 1;
% t = 2;
% cmin = GT{i}(t,1);
% rmin = GT{i}(t,2);
% width = GT{i}(t,3);
% height = GT{i}(t,4);
% 
% x2 = [cmin cmin        cmin+width  cmin+width cmin];
% y2 = [rmin rmin+height rmin+height rmin       rmin];
% 
% [xa, ya] = polybool('union', x1, y1, x2, y2);
% [xb, yb] = polybool('intersection', x1, y1, x2, y2);
% subplot(2, 2, 1)
% % patch(xa, ya, 1, 'FaceColor', 'r')
% axis equal, axis off, hold on
% plot(x1, y1, x2, y2, 'Color', 'k')
% title('Union')
% 
% subplot(2, 2, 2)
% % patch(xb, yb, 1, 'FaceColor', 'r')
% axis equal, axis off, hold on
% plot(x1, y1, x2, y2, 'Color', 'k')
% title('Intersection')