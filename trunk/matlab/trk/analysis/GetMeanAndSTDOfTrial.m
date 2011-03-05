function [MeanVals, STD] = GetMeanAndSTDOfTrial(TRIAL, functionHandle, vaargin)


max_val = -1000000;
min_val =  1000000;

nExperiments = length(TRIAL.EXPERIMENTS);

% Gets the responses for the function for each experiment
for nE = 1:nExperiments
   resp{nE} = GetValuesFromExperiment(...
        TRIAL.EXPERIMENTS(nE), functionHandle, vaargin);

   maxv = max(resp{nE});
   minv = min(resp{nE});
   if( maxv > max_val)
       max_val = maxv;
   end
   if( minv < min_val)
       min_val = minv;
   end
end

MeanVals = zeros(nExperiments, 1);
STD  = zeros(nExperiments, 1);
for i = 1:1:nExperiments
   MeanVals(i) = mean(resp{i}); 
   STD(i) = std(resp{i}); 
end