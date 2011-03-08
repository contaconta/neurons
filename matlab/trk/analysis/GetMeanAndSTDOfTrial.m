function [MeanVals, STD, perc] = GetMeanAndSTDOfTrial(TRIAL, display, functionHandle, varargin)


max_val = -1000000;
min_val =  1000000;

nExperiments = length(TRIAL.EXPERIMENTS);

% Gets the responses for the function for each experiment
for nE = 1:nExperiments
   resp{nE} = GetValuesFromExperiment(...
        TRIAL.EXPERIMENTS(nE), functionHandle, varargin{:});

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

% Computes the percentage over the experiment
controlExperimentNumber = TRIAL.ExperimentNames('Not_Targ');
meanControlExperiment   = mean(resp{controlExperimentNumber});

perc = zeros(nExperiments, 1);
for t = 1:length(MeanVals)
   perc(t) = length(find(resp{t} > meanControlExperiment)) / length(resp{t});
end


if(display)


hold on;
plot([1 length(MeanVals)], [meanControlExperiment, meanControlExperiment],'k','LineWidth', 3);


errorbar(1:length(MeanVals)', MeanVals, STD,'*','MarkerSize', 7,'LineWidth', 2);
vals = TRIAL.ExperimentNames.values;
keys = TRIAL.ExperimentNames.keys;
for t = 1:length(MeanVals)
   idx = find([vals{:}] == t);
   hold on;
   plot(t*ones(length(resp{t}),1) + randn(length(resp{t}),1)*0.05, resp{t}, 'k.');
   plot([t-0.4, t+0.4], [MeanVals(t), MeanVals(t)], 'r','LineWidth', 3);
   plot([t-0.25, t+0.25], [MeanVals(t)+STD(t), MeanVals(t)+STD(t)], 'g','LineWidth', 3);
   plot([t-0.25, t+0.25], [MeanVals(t)-STD(t), MeanVals(t)-STD(t)], 'g', 'LineWidth', 3);
end



g = str2num(get(gca,'YTickLabel'));

for t = 1:length(MeanVals)
   text( t, g(end-1), num2str(round(perc(t)*100)),'FontSize', 12,'rotation', 90);
end



for i = 1:1:length(MeanVals)
   cellkey = keys(find([vals{:}] == i));
   xticklabelcell{i} = cellkey{1};
end
set(gca, 'FontSize', 12);
set(gca, 'XTick', 1:1:length(MeanVals));
set(gca,'XTickLabel',xticklabelcell)
ylabel(func2str(functionHandle));
%ylabel('MeanCableLength');
h = rotateticklabel(gca, 90);
end



