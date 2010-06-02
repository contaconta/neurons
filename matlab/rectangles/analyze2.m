NLEARNERS = 15;
IMSIZE = [24 24];
folder = [pwd '/results/'];

figure;  hold on; %set(gca, 'XScale', 'log');
legstr = {}; plotid = 2;
fplocs = [1.1e-6:.1e-6:1e-5,1.1e-5:.1e-5:1e-4,1.1e-4:.1e-4:1e-3,1.1e-3:.1e-3:1e-2, 1.1e-2:.1e-2:1e-1, 1.1e-1:.1e-1:1];


prefix = 'VJ-';
d = dir([folder prefix '*']); TPvj = zeros(length(d), length(fplocs));
for i = 1:length(d)
    filename = d(i).name;
    load([folder filename]);
    [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
    TPvj(i,:) = interp1(FP/NN,TP/NP,fplocs);
end
save(['roc' num2str(NLEARNERS) '.mat'], 'TPvj', 'fplocs');

prefix = 'A2-';
d = dir([folder prefix '*']); TPa2 = zeros(length(d), length(fplocs));
for i = 1:length(d)
    filename = d(i).name;
    load([folder filename]);
    [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
    TPa2(i,:) = interp1(FP/NN,TP/NP,fplocs);
end
save(['roc' num2str(NLEARNERS) '.mat']);

prefix = 'A4-';
d = dir([folder prefix '*']); TPa4 = zeros(length(d), length(fplocs));
for i = 1:length(d)
    filename = d(i).name;
    load([folder filename]);
    [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
    TPa4(i,:) = interp1(FP/NN,TP/NP,fplocs);
end
save(['roc' num2str(NLEARNERS) '.mat']);

prefix = 'A8-';
d = dir([folder prefix '*']); TPa8 = zeros(length(d), length(fplocs));
for i = 1:length(d)
    filename = d(i).name;
    load([folder filename]);
    [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
    TPa8(i,:) = interp1(FP/NN,TP/NP,fplocs);
end
save(['roc' num2str(NLEARNERS) '.mat']);

prefix = 'A12-';
d = dir([folder prefix '*']); TPa12 = zeros(length(d), length(fplocs));
for i = 1:length(d)
    filename = d(i).name;
    load([folder filename]);
    [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
    TPa12(i,:) = interp1(FP/NN,TP/NP,fplocs);
end
save(['roc' num2str(NLEARNERS) '.mat']);

% prefix = '50-50';
% d = dir([folder prefix '*']); TP50 = zeros(length(d), length(fplocs));
% for i = 1:length(d)
%     filename = d(i).name;
%     load([folder filename]);
%     [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
%     TP50(i,:) = interp1(FP/NN,TP/NP,fplocs);
% end
% save(['roc' num2str(NLEARNERS) '.mat']);
% 
% prefix = '33-';
% d = dir([folder prefix '*']); TP33 = zeros(length(d), length(fplocs));
% for i = 1:length(d)
%     filename = d(i).name;
%     load([folder filename]);
%     [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
%     TP33(i,:) = interp1(FP/NN,TP/NP,fplocs);
% end
% save(['roc' num2str(NLEARNERS) '.mat']);
% 
% prefix = 'VJANORM-';
% d = dir([folder prefix '*']); TPvja = zeros(length(d), length(fplocs));
% for i = 1:length(d)
%     filename = d(i).name;
%     load([folder filename]);
%     [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
%     TPvja(i,:) = interp1(FP/NN,TP/NP,fplocs);
% end
% save(['roc' num2str(NLEARNERS) '.mat']);

plot(fplocs, mean(TPvj,1), 'k', 'LineWidth', 2);
plot(fplocs, mean(TPa2,1), 'b');
plot(fplocs, mean(TPa4,1), 'r');
plot(fplocs, mean(TPa8,1), 'g');
plot(fplocs, mean(TPa12,1), 'y');
plot(fplocs, mean(TP50,1), 'c', 'LineWidth', 2);
plot(fplocs, mean(TP33,1), 'm', 'LineWidth', 2);
plot(fplocs, mean(TPvja,1), 'b--', 'LineWidth', 2);

legend('VJ', 'A2', 'A4', 'A8', 'A12', '50-50', '33', 'ANORM');
title(['Area Normalized vs Viola-Jones ' num2str(NLEARNERS) ' learners, ' num2str(NP) ' (+) / ' num2str(NN) ' (-) examples.']);


keyboard; 

x = 1:50:length(fplocs);

ploterr(fplocs(x), mean(TPvj(:,x),1), [ ], std(TPvj(:,x),0,1), 'k.', 'logx');

ploterr(fplocs(x), mean(TPvj(:,x),1), [ ], std(TPvj(:,x),0,1), 'k.', 'logx');
ploterr(fplocs(x), mean(TPa2(:,x),1), [ ], std(TPa2(:,x),0,1), 'b.', 'logx');
ploterr(fplocs(x), mean(TPa4(:,x),1), [ ], std(TPa4(:,x),0,1), 'r.', 'logx');
ploterr(fplocs(x), mean(TPa8(:,x),1), [ ], std(TPa8(:,x),0,1), 'g.', 'logx');
ploterr(fplocs(x), mean(TPa12(:,x),1), [ ], std(TPa12(:,x),0,1), 'y.', 'logx');
ploterr(fplocs(x), mean(TP50(:,x),1), [ ], std(TPvj(:,x),0,1), 'c.', 'logx');
ploterr(fplocs(x), mean(TP33(:,x),1), [ ], std(TPvj(:,x),0,1), 'm.', 'logx');
ploterr(fplocs(x), mean(TPvja(:,x),1), [ ], std(TPvja(:,x),0,1), 'b.', 'logx');

errtp = a(1:50:length(fplocs));
shadedErrorBar(log10(fplocs), TPa12, {@mean,@std}, '-y',1);
shadedErrorBar(log10(fplocs), TPa8, {@mean,@std}, '-g',1);
shadedErrorBar(log10(fplocs), TPa4, {@mean,@std}, '-r',1);
shadedErrorBar(log10(fplocs), TPa2, {@mean,@std}, '-b',1);
shadedErrorBar(log10(fplocs), TPvj, {@mean,@std}, {'-k', 'Linewidth', 2} ,1);
set(gca, 'XTickLabel', {1e-6, 1e-5,1e-4,1e-3,1e-2,1e-1,1});
prettygraph();





% errfp = fplocs(1:50:length(fplocs));
% errtp = a(1:50:length(fplocs));
% errstd = std(TPvj(:,1:50:length(fplocs)),0,1);