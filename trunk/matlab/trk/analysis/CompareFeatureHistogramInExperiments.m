function CompareFeatureHistogramInExperiments(TRIAL, featureHandle, varargin)


nBins = 100;

max_val = -1000000;
min_val =  1000000;

nExperiments = length(TRIAL.EXPERIMENTS);

% Gets the responses for the function for each experiment
for nE = 1:nExperiments
   resp{nE} = GetValuesFromExperiment(...
        TRIAL.EXPERIMENTS(nE), featureHandle, varargin);
   maxv = max(resp{nE});
   minv = min(resp{nE});
   if( maxv > max_val)
       max_val = maxv;
   end
   if( minv < min_val)
       min_val = minv;
   end
end


X = min_val:(max_val - min_val)/nBins:max_val;
max_bin = 0;
nPoints = zeros(1, nExperiments);
for nE = 1:1:nExperiments
   n{nE} = hist(resp{nE}, X);
   nPoints(nE) = sum(n{nE});
   n{nE} = n{nE} ./ nPoints(nE);
   if(max(n{nE}) > max_bin)
      max_bin = max(n{nE}); 
   end
end

figure;
for i = 1:nExperiments
 ax(i) = subplot(1,nExperiments,i);
 bar(X, n{i});
 axis([min(X), max(X), 0, max_bin]);
 title([TRIAL.EXPERIMENTS(i).RUNS(1).GlobalMeasures.Label ' ' num2str(nPoints(i))]);
end
linkaxes(ax, 'xy');
