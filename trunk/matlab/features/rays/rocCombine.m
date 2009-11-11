rocFolder = '/osshare/DropBox/Dropbox/aurelien/roc/final/';

filename = 'allROCs.png';

figure; hold on;
grid on;


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

Cols = [0.2    0.2000    .75000;...
         .3    .3000    .3000;...
    0.0000    .8000    0.000;...
    0.8000    0.8000         0;...
    1.0000    0.000         0];

load([rocFolder 'HistHist.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
legnames{2} = 'Hist learned pairwise';
legnames{1} = 'Hist int diff pairwise';
plot(fpr{1}, tpr{1}, '--', 'Color', Cols(1,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, '-', 'Color', Cols(1,:), 'LineWidth', 2.5); hold on;

load([rocFolder 'SteerSteer.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0;  tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1; % corrections
legnames{4} = 'Steer learned pairwise';
legnames{3} = 'Steer int diff pairwise';
plot(fpr{1}, tpr{1}, '--', 'Color', Cols(2,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, '-', 'Color', Cols(2,:), 'LineWidth', 2.5); hold on;

load([rocFolder 'RayP1.mat']); 
fpr{1}(1) = 0; tpr{1}(1) = 0; tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
legnames{6} = 'Ray learned pairwise';
legnames{5} = 'Ray int diff pairwise';
plot(fpr{1}, tpr{1}, '--', 'Color', Cols(3,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, '-', 'Color', Cols(3,:), 'LineWidth', 2.5); hold on;

load([rocFolder 'HistRayP1.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0; tpr{1}(length(tpr{1}+1)) = tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;  % corrections
fpr{3}(1) = 0; tpr{3}(1) = 0; tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1;  % corrections
legnames{8} = 'Ray+Hist learned pairwise';
legnames{7} = 'Ray+Hist int diff pairwise';
plot(fpr{1}, tpr{1}, '--', 'Color', Cols(4,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, '-', 'Color', Cols(4,:), 'LineWidth', 2.5); hold on;

load([rocFolder 'HistRaySteerP1.mat']);
fpr{1}(1) = 0; tpr{1}(1) = 0;  tpr{1}(length(tpr{1}+1)) = tpr{1}(length(tpr{1})); fpr{1}( length(fpr{1}+1)) = 1;% corrections
fpr{3}(1) = 0; tpr{3}(1) = 0;  tpr{3}(length(tpr{3}+1)) = tpr{3}(length(tpr{3})); fpr{3}( length(fpr{3}+1)) = 1; % corrections
legnames{10} = 'Ray+Hist+Steer learned pairwise';
legnames{9} = 'Ray+Hist+Steer int diff pairwise';
plot(fpr{1}, tpr{1}, '--', 'Color', Cols(5,:), 'LineWidth', .75); hold on;
plot(fpr{3}, tpr{3}, '-', 'Color', Cols(5,:), 'LineWidth', 2.5); hold on;

legend(legnames, 'Location', 'SouthEast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');

axis([0 0.14 0 1]);

print(gcf, '-dpng', '-r150', [rocFolder filename '.png']);
%save([rocFolder filename '.mat'], 'tpr', 'fpr', 'd');