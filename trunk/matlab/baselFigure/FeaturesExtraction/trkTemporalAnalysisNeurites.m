function ListOfNeuriteTracksAssociatedToCellTrack = trkTemporalAnalysisNeurites(ListOfNeuriteTracksAssociatedToCellTrack, neuriteFieldsToAnalyse)

for i = 1:length(neuriteFieldsToAnalyse)
    fieldArray = [];
    timeArray  = [];
    NewListOfNeuritesTracked = [];
    for k = 1:length(ListOfNeuriteTracksAssociatedToCellTrack)
        neuritesTrack = ListOfNeuriteTracksAssociatedToCellTrack(k);
        for j =1:length(neuritesTrack.Neurites)
            fieldArray = [fieldArray getfield(neuritesTrack.Neurites(j), neuriteFieldsToAnalyse{i})];%#ok
            timeArray  = [timeArray  getfield(neuritesTrack.Neurites(j), 'Time')];%#ok
        end
        fieldArrayName  = [neuriteFieldsToAnalyse{i} 'Array'];
        neuritesTrack    = setfield(neuritesTrack, fieldArrayName, fieldArray);%#ok

        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(fieldArray);
        neuritesTrack.([neuriteFieldsToAnalyse{i} 'expContractVector'])     = expContractVector;
        neuritesTrack.([neuriteFieldsToAnalyse{i} 'timeExpanding'])         = timeExpanding;
        neuritesTrack.([neuriteFieldsToAnalyse{i} 'timeContracting'])       = timeContracting;
        neuritesTrack.([neuriteFieldsToAnalyse{i} 'numberInflexionPoints']) = numberInflexionPoints;
        neuritesTrack.([neuriteFieldsToAnalyse{i} 'freqExpansion'])         = freqExpansion;
        neuritesTrack.(['Mean' neuriteFieldsToAnalyse{i}])                  = mean(fieldArray);
        deltaFieldArray = fieldArray(2:end) - fieldArray(1:end-1);
        deltaTimeArray  = timeArray(2:end)  - timeArray(1:end-1);
        deltaFieldArray = deltaFieldArray ./ deltaTimeArray;

        deltaFieldName  = ['Delta' neuriteFieldsToAnalyse{i}];
        for t =1:length(neuritesTrack.Neurites)-1
            neuritesTrack.Neurites(t+1).(deltaFieldName)    = deltaFieldArray(t);
        end
        
        NewListOfNeuritesTracked = [NewListOfNeuritesTracked neuritesTrack];%#ok
    end
    ListOfNeuriteTracksAssociatedToCellTrack = NewListOfNeuritesTracked;
end