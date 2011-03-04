
close all;
clear all;

%% Loads the experiments

disp('... loading experiments');

TRIALS(1) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '14-11-2010_%03i.mat');
% TRIALS(2) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '15-11-2010_%03i.mat');
% TRIALS(3) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '16-11-2010_%03i.mat');
% TRIALS(4) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '17-11-2010_%03i.mat');

disp('... pfew');



%% Creats an statistical function handle
statistic = @NeuriteLength;
statistic = @BranchingPointsPerNeurite;
st_arg = 0;


%% Computes the mean value of the function among the experiments of the trial
[Mean, STD] = GetMeanAndSTDOfTrial(TRIALS(1), statistic, st_arg);
errorbar(1:length(Mean)', Mean, STD);


%% Plots the histogram of the values on the trial
vals = GetValuesFromExperiment(TRIALS(1).EXPERIMENTS(1), statistic, st_arg);
hist(vals, 100);





%% There is still a lot of work to reach this point

% %% Computes the confusion matrix of a trial using the given handle
% ConfMatrix = GetConfusionMatrixOfTrial(TRIALS(1), statistic, st_arg);
% figure;
% imagesc(ConfMatrix);
% title('Confusion Matrix of the Experiments of the Trial');
% 
% %% Computes the confusion matrix of an experiment across trials
% nExperiment = 1;
% ConfMatrix = GetConfusionMatrixOfExperimentAcrossTrials...
%     (TRIALS, nExperiment, statistic, st_arg);
% figure;
% imagesc(ConfMatrix);
% title('Confusion Matrix of the Experiments of the Trial');
% 
% %% Gets the confusion matrix of the runs of an experiment
% ConfMatrix = GetConfusionMatrixOfExperiment...
%     (TRIALS(1).EXPERIMENTS(1), statistic, st_arg);
% figure;
% imagesc(ConfMatrix);
% title('Confusion Matrix of the Runs of an Experiment');
% 









%% Tests the contents of the experiment
% figure;
% for t = 1:1:length(EXP{1}.Dlist)
%     renderData(EXP{1}, t);
%     pause(0.1);
% end
% 
% 
% %% Follows the life of a neuron
% renderNeuronTrack(EXP{1}, 12)
