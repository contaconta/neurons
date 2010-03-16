rocFolder = '/osshare/DropBox/Dropbox/aurelien/roc/final/';

filename = 'allROCs.png';

figure; hold on;


% Cols = [1 0 0;...
%         .7 0 0; ...
%         0 1 0; ...
%         0 .7 0; ...
%         0 0 1; ...
%         0 0 .7;...
%         .4 .4 .4;...
%         .1 .1 .1;...
%         1 1 0;...
%         .7 .7 0];

%Cols = .75*jet(5);

Cols = [0.3137    0.3137    .3137;...
    .2902 .6039 .2902;...
    .7765 .2510 .2510;...
    .3098 .4314 .8784];

% Cols = [0.2    0.2000    .75000;...
%          .3    .3000    .3000;...
%     0.0000    .8000    0.000;...
%     0.8000    0.8000         0;...
%     1.0000    0.000         0];


load([rocFolder 'HistHist.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
%legnames{2} = 'Learned Histogram';
%legnames{1} = 'Standard Histogram';
legnames{2} = 'Learned-f^{Hist}';
legnames{1} = 'Standard-f^{Hist}';
plot(fpr{1}, tpr{1}, 's--', 'Color', Cols(1,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 's-', 'Color', Cols(1,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(1,:)); hold on;

load([rocFolder 'SteerSteer.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0;  tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1; % corrections
legnames{4} = 'Learned Rotational';
legnames{3} = 'Standard Rotational';
plot(fpr{1}, tpr{1}, 'v--', 'Color', Cols(2,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 'v-', 'Color', Cols(2,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(2,:)); hold on;

load([rocFolder 'HistRayP1.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0; tpr{1}(length(tpr{1}+1)) = tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
legnames{8} = 'Learned Rays';
legnames{7} = 'Standard Rays';
plot(fpr{1}, tpr{1}, 'x--', 'Color', Cols(3,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 'x-', 'Color', Cols(3,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(3,:)); hold on;

load([rocFolder 'RayHistSteerRayHistSteer.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1}+1)) = tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;% corrections
fpr{3}(1) = 0; tpr{3}(1) = 0;  tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1; % corrections
legnames{8} = 'Learned Combo';
legnames{7} = 'Standard Combo';
plot(fpr{1}, tpr{1}, 'o--', 'Color', Cols(4,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 'o-', 'Color', Cols(4,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(4,:)); hold on;


line([0 1], [.1 .1], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.2 .2], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.3 .3], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.4 .4], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.5 .5], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.6 .6], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.7 .7], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.8 .8], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [.9 .9], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 1], [1 1], 'Color', [1 1 1], 'LineWidth', 1.75);
line([0 0], [0 1], 'Color', [1 1 1], 'LineWidth', 2);

load([rocFolder 'HistHist.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
legnames{2} = 'Learned Histogram';
legnames{1} = 'Standard Histogram';
legnames{2} = 'Learned-f^{Hist}';
legnames{1} = 'Standard-f^{Hist}';
plot(fpr{1}, tpr{1}, 's--', 'Color', Cols(1,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 's-', 'Color', Cols(1,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(1,:)); hold on;

load([rocFolder 'SteerSteer.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0;  tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1; % corrections
legnames{4} = 'Learned Rotational';
legnames{3} = 'Standard Rotational';
plot(fpr{1}, tpr{1}, 'v--', 'Color', Cols(2,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 'v-', 'Color', Cols(2,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(2,:)); hold on;

% load([rocFolder 'RayRay.mat']); 
% fpr{1}(1) = 0; tpr{1}(1) = 0; tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
% fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
% legnames{6} = 'Ray learned pairwise';
% legnames{5} = 'Ray int diff pairwise';
% plot(fpr{1}, tpr{1}, '--', 'Color', Cols(3,:), 'LineWidth', .75); hold on;
% plot(fpr{3}, tpr{3}, '-', 'Color', Cols(3,:), 'LineWidth', 2.5); hold on;

load([rocFolder 'HistRayP1.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0; tpr{1}(length(tpr{1}+1)) = tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
legnames{6} = 'Learned Rays';
legnames{5} = 'Standard Rays';
plot(fpr{1}, tpr{1}, 'x--', 'Color', Cols(3,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 'x-', 'Color', Cols(3,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(3,:)); hold on;

load([rocFolder 'RayHistSteerRayHistSteer.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1}+1)) = tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;% corrections
fpr{3}(1) = 0; tpr{3}(1) = 0;  tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1; % corrections
legnames{8} = 'Learned Combo';
legnames{7} = 'Standard Combo';
plot(fpr{1}, tpr{1}, 'o--', 'Color', Cols(4,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, 'o-', 'Color', Cols(4,:), 'LineWidth', 2.5, 'MarkerFaceColor', Cols(4,:)); hold on;

set(gca, 'Color', [.8 .8 .8]);
set(gcf, 'Color', [1  1 1]);

legend(legnames, 'Location', 'SouthEast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');

axis([0 0.08 0 1]);

print(gcf, '-dpng', '-r150', [rocFolder filename '.png']);
%save([rocFolder filename '.mat'], 'tpr', 'fpr', 'd');