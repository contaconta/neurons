
close all;
clear all;

%% Loads the experiments

disp('... loading experiments');
addpath('../');
dir = '/net/cvlabfiler1/home/ksmith/Basel/PostResults/';

TRIALS(1) = LoadTrial(dir, '14-11-2010_%03i.mat');
TRIALS(2) = LoadTrial(dir, '15-11-2010_%03i.mat');
TRIALS(3) = LoadTrial(dir, '16-11-2010_%03i.mat');
% TRIALS(4) = LoadTrial(dir, '17-11-2010_%03i.mat');

disp('... pfew');

%%



%% FIGURE OF THE NEURITE LENGTHS
LookOnlyAtHappyNeurons = 1;
statistic = @ffMajorAxisLengthNeurite;
[Mean, STD] = GetMeanAndSTDOfExperimentsAcrossTrials(TRIALS(1:3), statistic, LookOnlyAtHappyNeurons);

%% FIGURE OF THE NUMBER OF NEURITES
LookOnlyAtHappyNeurons = 1;
statistic = @fnMeanNumTrackedNeurites;
[Mean, STD] = GetMeanAndSTDOfExperimentsAcrossTrials(TRIALS(1:3), statistic, LookOnlyAtHappyNeurons);

%% Look for the FILOPODIA
LookOnlyAtHappyNeurons = 1;
% statistic = @faMeanFiloPercentNeurite; % Not working
% statistic = @faMeanFiloCountNeurite; % WORKING
statistic = @faMeanFiloCableLengthNeurite;
[Mean, STD] = GetMeanAndSTDOfExperimentsAcrossTrials(TRIALS(1:3), statistic, LookOnlyAtHappyNeurons);

%% Test to see which functional works best
close all;
clc;
display = 0;
LookOnlyAtHappyNeurons = 1;
nBins = 100;
statistics = {};
statistics{1} = @faMaxDistToSomaExtremeNeurite;
statistics{2} = @faMaxDistToSomaMedianNeurite;
statistics{3} = @faMaxDistToSomaStandDevNeurite;
statistics{4} = @faMaxDistToSomaMeanNeurite;
statistics{5} = @faMaxMajorAxisLengthNeurite;
statistics{6} = @faMaxTotalCableLengthNeurite;
statistics{7} = @ffDistToSomaExtremeNeurite;
statistics{8} = @ffDistToSomaMeanNeurite;
statistics{9} = @ffDistToSomaMedianNeurite;
statistics{10} = @ffDistToSomaStandDevNeurite;
statistics{11} = @ffMajorAxisLengthNeurite;
statistics{12} = @ffTotalCableLengthNeurite;

for trial = 1:1
disp(['=============== TRIAL ' num2str(trial) '===============']);
for nE = 1:12
   [Mean, STD, perc] = GetMeanAndSTDOfTrial(TRIALS(trial), display, statistics{nE}, LookOnlyAtHappyNeurons);
   distance = ComputeDistanceToDesiredOtuput(perc);
   disp(['Distance with measurement ' func2str(statistics{nE}) ' = ' num2str(distance)]);
end
end


%% Creats an statistical function handle
% statistic = @ffNeuriteLength;
%statistic = @ffMeanGreenIntensitySomata;
% statistic = @ffNumberOfNeuritesPerDetection;
% statistic = @fnMeanNeuriteLength;

% statistic = @ffAreaSomata;
% statistic = @ffAreaNuclei;
% statistic = @ffTravelDistanceNuclei;



% statistic = @ffBranchingPointsPerNeurite;
% statistic = @ffEndPointsPerNeurite;


% statistic = @ffNumberOfNeuritesPerDetection;
% statistic = @ffNeuriteLength;

% statistic = @fnDeltaNeuronLength;
% statistic = @fnMeanBranchingPoints;
% statistic = @fnMeanEndPoints;

% statistic = @ffNeuronLength;
% statistic = @fnMeanNeuriteLength;
% statistic = @fnMeanNeuronLength;

% statistic = @fnNucleusTimeExpanding;
% statistic = @fnMeanKevinTotalCableLength;
% statistic = @fnGermanTotalCableLengthFreqExpansion;
statistic = @fnMaxKevinTotalCableLength;
% statistic = @fnMeanNumTrackedNeurites;
% statistic = @ffBranchCountNeurite;





%% %% Computes the mean value of the function among the experiments of all trial
close all;
clc;
nBins = 100;

CompareFeatureHistogramInExperimentsAcrossTrials(TRIALS, nBins, statistic, LookOnlyAtHappyNeurons);



%% TRY TO FIND A FIGURE FOR STEADY GROWTH
% fail
LookOnlyAtHappyNeurons = 1;
statistic = @fnGermanTotalCableLengthFreqExpansion;
statistic = @fnGermanTotalCableLengthTimeExpanding;
[Mean, STD] = GetMeanAndSTDOfExperimentsAcrossTrials(TRIALS(1:3), statistic, LookOnlyAtHappyNeurons);   



%%

figure;
errorbar(1:length(Mean)', Mean, STD);
vals = TRIALS(1).ExperimentNames.values;
keys = TRIALS(1).ExperimentNames.keys;
for t = 1:length(Mean)
    idx = find([vals{:}] == t);
   text( t-0.5, Mean(t), keys(idx));
end




%% Computes the mean value of the function among the experiments of the trial
close all;
LookOnlyAtHappyNeurons = 1;
nBins = 100;

CompareFeatureHistogramInExperiments(TRIALS(1), nBins, statistic, LookOnlyAtHappyNeurons);


%% close all;
[Mean, STD, perc] = GetMeanAndSTDOfTrial(TRIALS(2), 1, statistics{11}, LookOnlyAtHappyNeurons);

%%
nBins = 100;
% statistic = @faMaxMajorAxisLengthNeurite;
statistic = @faMaxDistToSomaExtremeNeurite;
LookOnlyAtHappyNeurons = 1;

yLimMax = 0;
yLimMin = 50000;

figure;
for t = 1:4
   ax(t) = subplot(2,2,t);
   [Mean, STD] = GetMeanAndSTDOfTrial(TRIALS(t), 1, statistics{11}, LookOnlyAtHappyNeurons);
   title(['TRIAL ' num2str(t)]);
   YL = get(gca, 'YLim');
   if (YL(1) < yLimMin)
       yLimMin = YL(1);
   end
   if(YL(2) > yLimMax)
       yLimMax = YL(2);
   end
end
linkaxes(ax, 'xy');

for t = 1:4
    subplot(2,2,t);
    set(gca, 'YLim', [yLimMin, yLimMax]);
end


%% Sanity check for the mean and variance
LookOnlyAtHappyNeurons = 0;
nBins = 100;
CompareFeatureHistogramInExperiments(TRIALS(1), nBins, statistic, LookOnlyAtHappyNeurons);
[m,s]  = GetMeanAndSTDOfTrial(TRIALS(1), statistic, LookOnlyAtHappyNeurons);


subplot(1,2,1);
hold on;
% gaussian fit
x = -100:.1:3000;
plot(x, 0.08*exp( - (x-m(1)).*(x-m(1))/(2*s(1)*s(1))),'r');
subplot(1,2,2);
hold on;
% adjust height accordingly
x = 0:.1:1000;
lmbd = 1/m(2);
plot(x, 0.08*exp( - (x-m(2)).*(x-m(2))/(2*s(2)*s(2))),'r');
%plot(x, lmbd*exp(-lmbd*x),'r');




%% Compares the statistic in all neurons or only happy ones
close all;
nBins = 100;
LookOnlyAtHappyNeurons = 1;
[fig1, s1] = CompareFeatureHistogramInExperiments(TRIALS(1), nBins, statistic, LookOnlyAtHappyNeurons);
LookOnlyAtHappyNeurons = 0;
[fig2, s2] = CompareFeatureHistogramInExperiments(TRIALS(1), nBins, statistic, LookOnlyAtHappyNeurons);
LookOnlyAtHappyNeurons = 1;

s3 = [min(s1(1), s2(1)), max(s1(2), s2(2)), min(s1(3), s2(3)), max(s1(4), s2(4))];
figure(fig1);
axis(s3);
figure(fig2);
axis(s3);



%% Check for the labels of the experiment
vals = TRIALS(1).ExperimentNames.values;
keys = TRIALS(1).ExperimentNames.keys;

clc;
for nExperiment = 1:length(TRIALS(1).EXPERIMENTS)
    idx = find([vals{:}] == nExperiment);
    nameExp = keys{idx};
    if ~isequal(nExperiment, vals{idx})
       disp(['Error between experiment value and map at nExperiment = ' num2str(nExperiment) ' idx=' num2str(idx)]); 
    end
    for nRun = 1:length(TRIALS(1).EXPERIMENTS(nExperiment).RUNS)
        if ~isequal(nameExp, TRIALS(1).EXPERIMENTS(nExperiment).RUNS(nRun).GlobalMeasures.Label)
            disp(['nRun= ' num2str(nRun)]);
            disp(['Error between RunName = ' TRIALS(1).EXPERIMENTS(nExperiment).RUNS(nRun).GlobalMeasures.Label ' ; nExperiment = ' nameExp]); 
        end
        
    end
end







%% Plots the histogram of the values on the trial
% vals = GetValuesFromExperiment(TRIALS(1).EXPERIMENTS(2), statistic, st_arg);
% figure
% hist(vals, 100);





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
