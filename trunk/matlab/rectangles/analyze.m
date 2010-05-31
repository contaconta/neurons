
NLEARNERS = 1500;

IMSIZE = [24 24];

figure(1);  hold on; set(gca, 'XScale', 'log');
folder = [pwd '/results/'];
legstr = {}; plotid = 2;


filename = 'VJ-Restart-cvlabpc47-May292010-175511.mat';
load([folder filename]);
%figure(1); [TPvj FPvj] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
figure(1); [TPvj FPvj NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
plot(FPvj/NN, TPvj/NP, 'k', 'LineWidth', 2);
% figure(plotid); plot_classifier_composition(CLASSIFIER); plotid = plotid+1;
% figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); plotid = plotid+1;
legstr{length(legstr)+1} = 'Viola-Jones'; drawnow; pause(0.001);

% load results/VJ-Rank2-Restart-cvlabpc5-May292010-184423.mat
% figure(1); [TPvj2 FPvj2] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
% plot(FPvj2/NN, TPvj2/NP, 'b', 'LineWidth', 2);
% % figure(plotid); plot_classifier_composition(CLASSIFIER); plotid = plotid+1;
% % figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); plotid = plotid+1;
% legstr{length(legstr)+1} = 'VJ Rank 2'; drawnow; pause(0.001);

filename = 'A2-cvlabpc47-May292010-190042.mat';
load([folder filename]);
%figure(1); [TPa2 FPa2] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
figure(1); [TPa2 FPa2 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
plot(FPa2/NN, TPa2/NP, 'b');
% figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 2'); plotid = plotid+1;
% figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 2'); plotid = plotid+1;
legstr{length(legstr)+1} = 'ANorm Rank 2'; drawnow; pause(0.001);

filename = 'A4-cvlabpc47-May292010-181619.mat';
load([folder filename]);
%figure(1); [TPa4 FPa4] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
figure(1); [TPa4 FPa4 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
plot(FPa4/NN, TPa4/NP, 'r');
% figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 4'); plotid = plotid+1;
% figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 4'); plotid = plotid+1;
legstr{length(legstr)+1} = 'ANorm Rank 4';  drawnow; pause(0.001);

filename = 'A8-cvlabpc2-May292010-182306.mat';
load([folder filename]);
%figure(1); [TPa8 FPa8] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
figure(1); [TPa8 FPa8 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
plot(FPa8/NN, TPa8/NP, 'm');
% figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 8'); plotid = plotid+1;
% figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 8'); plotid = plotid+1;
legstr{length(legstr)+1} = 'ANorm Rank 8';  drawnow; pause(0.001);

% filename = 'A12-cvlabpc7-May292010-182721.mat';
% load([folder filename]);
% % load results/A12-cvlabpc7-May292010-182721.mat
% figure(1); [TPa12 FPa12 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
% % figure(1); [TPa12 FPa12] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
% plot(FPa12/NN, TPa12/NP, 'c');
% % % figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 12'); plotid = plotid+1;
% % % figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 12'); plotid = plotid+1;
% legstr{length(legstr)+1} = 'ANorm Rank 12';  drawnow; pause(0.001);

filename = 'SIMPLE2-cvlabpc7-May292010-183152.mat';
load([folder filename]);
figure(1); [TPs2 FPs2 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
%figure(1); [TPs2 FPs2] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
plot(FPs2/NN, TPs2/NP, 'b:');
%figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 2'); plotid = plotid+1;
%figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 2'); plotid = plotid+1;
legstr{length(legstr)+1} = 'Simple Rank 2';  drawnow; pause(0.001);

filename = 'SIMPLE4-cvlabpc47-May292010-183501.mat';
load([folder filename]);
figure(1); [TPs4 FPs4 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
%figure(1); [TPs4 FPs4] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
plot(FPs4/NN, TPs4/NP, 'r:');
%figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 4'); plotid = plotid+1;
%figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 4'); plotid = plotid+1;
legstr{length(legstr)+1} = 'Simple Rank 4';  drawnow; pause(0.001);

% filename = 'SIMPLE8-cvlabpc7-May292010-183656.mat';
% load([folder filename]);
% figure(1); [TPs8 FPs8 NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
% %figure(1); [TPs8 FPs8] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
% plot(FPs8/NN, TPs8/NP, 'm:');
% %figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 8'); plotid = plotid+1;
% %figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 8'); plotid = plotid+1;
% legstr{length(legstr)+1} = 'Simple Rank 8';  drawnow; pause(0.001);
%
% load results/SIMPLE12-cvlabpc2-May292010-183846.mat
% figure(1); [TPs12 FPs12] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
% plot(FPs12/NN, TPs12/NP, 'c:');
% %figure(plotid); plot_classifier_composition(CLASSIFIER); title('Area Norm Rank 12'); plotid = plotid+1;
% %figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Area Norm Rank 12'); plotid = plotid+1;
% legstr{length(legstr)+1} = 'Simple Rank 12';  drawnow; pause(0.001);

% load results/R2-cvlabpc7-May252010-213327.mat
% figure(1); [TPa12 FPa12] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
% plot(FPa12/NN, TPa12/NP, 'y', 'Linewidth', 2);
% % figure(plotid); plot_classifier_composition(CLASSIFIER); title('Non-Norm Rank 2'); plotid = plotid+1;
% % figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Non-Norm Rank 2'); plotid = plotid+1;
% legstr{length(legstr)+1} = 'NoNorm Rank 2';  drawnow; pause(0.001);
% 
% load results/R8-cvlabpc47-May252010-213714.mat
% figure(1); [TPa12 FPa12] = roc_evaluate(CLASSIFIER, NLEARNERS, TESTD, TESTL);
% plot(FPa12/NN, TPa12/NP, 'g', 'Linewidth', 2);
% % figure(plotid); plot_classifier_composition(CLASSIFIER); title('Non-Norm Rank 2'); plotid = plotid+1;
% % figure(plotid); plot_100_learners(CLASSIFIER, [24 24]); title('Non-Norm Rank 2'); plotid = plotid+1;
% legstr{length(legstr)+1} = 'NoNorm Rank 8';  drawnow; pause(0.001);

figure(1); title(['Area Normalized vs Viola-Jones ' num2str(NLEARNERS) ' learners, ' num2str(NP) ' (+) / ' num2str(NN) ' (-) examples.']);
%legstr = {'Viola-Jones', 'Rank 2', 'Rank 4', 'Rank 8', 'Rank 12', 'Non Rank 2'};
legend(legstr);