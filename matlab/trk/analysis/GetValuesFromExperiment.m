function values = GetValuesFromExperiment( Experiment, functionHandle, varargin)

% values = EXTRACTSTATISTICALVALUESFROMEXPERIMENT
%             ( experiment,functionHandle, vaargin)
%
% evaluates the function defined by functionHandle in all the runs of the
% experiment passing the variable arguments to the function


values = [];
for i = 1:1:length(Experiment.RUNS)
   valtmp = functionHandle(Experiment.RUNS(i), varargin{:});
   values = [values; valtmp];
end







