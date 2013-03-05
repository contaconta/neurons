function currentTrack = trkTemporalAnalysis(currentTrack, fieldsToAnalyse)

for i = 1:length(fieldsToAnalyse)
    fieldArray = [];
    timeArray  = [];
    for j =1:length(currentTrack.TimeStep)
        fieldArray = [fieldArray getfield(currentTrack.TimeStep(j), fieldsToAnalyse{i})];%#ok
        timeArray  = [timeArray  getfield(currentTrack.TimeStep(j), 'Time')];%#ok
    end
    fieldArrayName  = [fieldsToAnalyse{i} 'Array'];
    currentTrack    = setfield(currentTrack, fieldArrayName, fieldArray);%#ok
    
    [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(fieldArray);
    currentTrack.([fieldsToAnalyse{i} 'expContractVector'])     = expContractVector;
    currentTrack.([fieldsToAnalyse{i} 'timeExpanding'])         = timeExpanding;
    currentTrack.([fieldsToAnalyse{i} 'timeContracting'])       = timeContracting;
    currentTrack.([fieldsToAnalyse{i} 'numberInflexionPoints']) = numberInflexionPoints;
    currentTrack.([fieldsToAnalyse{i} 'freqExpansion'])         = freqExpansion;
    currentTrack.(['Mean' fieldsToAnalyse{i}])                  = mean(fieldArray);
    deltaFieldArray = fieldArray(2:end) - fieldArray(1:end-1);
    deltaTimeArray  = timeArray(2:end)  - timeArray(1:end-1);
    deltaFieldArray = deltaFieldArray ./ deltaTimeArray;
    
    deltaFieldName  = ['Delta' fieldsToAnalyse{i}];
    for t =1:length(currentTrack.TimeStep)-1
        currentTrack.TimeStep(t+1).(deltaFieldName)    = deltaFieldArray(t);
    end
end