function values = ExtractStatisticalValuesFromExperimentAcrossTrials ...
    ( TRIALS, ExperimentNumber, functionHandle, varargin)

values = [];

% Finds the name of the experiment
keyValues  = TRIALS(1).ExperimentNames.keys;
keyIndexes = TRIALS(1).ExperimentNames.values;
idx = find([keyIndexes{:}] == ExperimentNumber);
experimentName = keyValues{idx};


for i = 1:1:length(TRIALS)
   experimentNumberInTrial = TRIALS(i).ExperimentNames(experimentName);
   valtmp = GetValuesFromExperiment(...
        TRIALS(i).EXPERIMENTS(experimentNumberInTrial), functionHandle, varargin{:});
   values = [values; valtmp];
end