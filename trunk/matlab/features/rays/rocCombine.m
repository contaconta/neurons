rocFolder = '/osshare/DropBox/Dropbox/aurelien/roc/3class/';

filename = 'allROCs.png';

figure; hold on;
grid on;


Cols = [1 0 0;...
        .7 0 0; ...
        0 1 0; ...
        0 .7 0; ...
        0 0 1; ...
        0 0 .7;...
        .4 .4 .4;...
        .1 .1 .1];


load([rocFolder 'hist.mat']);
legnames{2} = 'Hist learned pairwise';
legnames{1} = 'Hist int diff pairwise';
plot(fpr{1}, tpr{1}, '.--', 'Color', Cols(1,:), 'LineWidth', 1); hold on;
plot(fpr{3}, tpr{3}, '.-', 'Color', Cols(2,:), 'LineWidth', 2); hold on;

load([rocFolder 'steer.mat']);
legnames{4} = 'Steer learned pairwise';
legnames{3} = 'Steer int diff pairwise';
plot(fpr{1}, tpr{1}, '.--', 'Color', Cols(3,:), 'LineWidth', 1); hold on;
plot(fpr{3}, tpr{3}, '.-', 'Color', Cols(4,:), 'LineWidth', 2); hold on;

load([rocFolder 'Rays.mat']);
legnames{6} = 'Ray learned pairwise';
legnames{5} = 'Ray int diff pairwise';
plot(fpr{1}, tpr{1}, '.--', 'Color', Cols(5,:), 'LineWidth', 1); hold on;
plot(fpr{3}, tpr{3}, '.-', 'Color', Cols(6,:), 'LineWidth', 2); hold on;

load([rocFolder 'raysHist.mat']);
legnames{8} = 'Ray+Hist learned pairwise';
legnames{7} = 'Ray+Hist int diff pairwise';
plot(fpr{1}, tpr{1}, '.--', 'Color', Cols(7,:), 'LineWidth', 1); hold on;
plot(fpr{3}, tpr{3}, '.-', 'Color', Cols(8,:), 'LineWidth', 2); hold on;


legend(legnames, 'Location', 'SouthEast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');

axis([0 0.2 0 1]);

print(gcf, '-dpng', '-r150', [rocFolder filename '.png']);
%save([rocFolder filename '.mat'], 'tpr', 'fpr', 'd');