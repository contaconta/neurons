function [currentTrackedCellTimeStep] = trkComputeQuantilesAndMean(currentTrackedCellTimeStep, fieldsToQuantile, quantilesList)

for l =1:length(fieldsToQuantile)
    quant = quantile(getfield(currentTrackedCellTimeStep, fieldsToQuantile{l}), quantilesList);%#ok
    for m =1:length(quantilesList)
        newFieldName                 = [fieldsToQuantile{l} '_q_' int2str(100*quantilesList(m))];
        currentTrackedCellTimeStep   = setfield(currentTrackedCellTimeStep, newFieldName, quant(m));%#ok
    end
    meanFieldName  = [fieldsToQuantile{l} 'Mean'];
    currentTrackedCellTimeStep       = setfield(currentTrackedCellTimeStep, meanFieldName, mean(getfield(currentTrackedCellTimeStep, fieldsToQuantile{l})));%#ok 
end