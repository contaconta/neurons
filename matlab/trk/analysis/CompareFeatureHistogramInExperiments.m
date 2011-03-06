function [fig,span] = CompareFeatureHistogramInExperiments(TRIAL, nBins, featureHandle, varargin)


max_val = -1000000;
min_val =  1000000;
minDetectLength = 1e10;

nExperiments = length(TRIAL.EXPERIMENTS);

% Gets the responses for the function for each experiment
for nE = 1:nExperiments
   resp{nE} = GetValuesFromExperiment(...
        TRIAL.EXPERIMENTS(nE), featureHandle, varargin{:});
   maxv = max(resp{nE});
   minv = min(resp{nE});
   if(length(resp{nE}) < minDetectLength)
       minDetectLength = length(resp{nE});
   end
   if( maxv > max_val)
       max_val = maxv;
   end
   if( minv < min_val)
       min_val = minv;
   end
end


x_step = (max_val - min_val)/nBins;
X = min_val:x_step:max_val;
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

span = [min(X)-x_step*5, max(X)+x_step*5, 0, max_bin]; 

fig = figure;
for i = 1:nExperiments
 ax(i) = subplot(3,ceil(nExperiments/3),i);
 bar(X, n{i});
 grid on;
 axis(span);
 title([TRIAL.EXPERIMENTS(i).RUNS(1).GlobalMeasures.Label ' ' num2str(nPoints(i))]);
end
linkaxes(ax, 'xy');
