function ConfMatrix = GetConfusionMatrixOfExperiment(EXPERIMENT, functionHandle, vaargin)


max_val = -1000000;
min_val =  1000000;

nRuns = length(EXPERIMENT.RUNS);

% Gets the responses for the function for each experiment
for nE = 1:nRuns
   resp{nE} = functionHandle(EXPERIMENT.RUNS(nE), vaargin);

   maxv = max(resp{nE});
   minv = min(resp{nE});
   if( maxv > max_val)
       max_val = maxv;
   end
   if( minv < min_val)
       min_val = minv;
   end
end

%% Computes the matrix
nBins = 100;
X = min_val:(max_val - min_val)/nBins:max_val;
for nE = 1:1:nRuns
   n{nE} = hist(resp{nE}, X);
   n{nE} = n{nE} ./ sum(n{nE});
end



%% Finally computes the matrix;
ConfMatrix = zeros(length(resp));
for i = 1:1:length(resp)
    for j = i+1:1:length(resp)
        ConfMatrix(i,j) = emdDistance(n{i}, n{j});
        ConfMatrix(j,i) = ConfMatrix(i,j);
    end
end

