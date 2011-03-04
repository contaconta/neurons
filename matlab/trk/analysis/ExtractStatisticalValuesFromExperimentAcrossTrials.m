function values = ExtractStatisticalValuesFromExperimentAcrossTrials ...
    ( TRIALS, ExperimentNumber, functionHandle, vaargin)

values = [];
for i = 1:1:length(TRIALS)
   Experiment = TRIALS(i).EXPERIMENTS(ExperimentNumber);
   valtmp = ExtractStatisticalValuesFromExperiment( Experiment, functionHandle, vaargin);
   values = [values; valtmp];
end