function [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(v)


expContractVector = zeros(size(v));

numberInflexionPoints = 0;
timeExpanding = 0;
timeContracting = 0;

for i = 1:length(v)-1
    if(v(i+1) > v(i))
        expContractVector(i) = 1;
        timeExpanding = timeExpanding + 1;
    else
        expContractVector(i) = -1; 
        timeContracting = timeContracting + 1;  
    end
end

expContractVector(end) = expContractVector(end-1);
if(expContractVector(end) == 1)
    timeExpanding = timeExpanding+1;
else
    timeContracting = timeContracting + 1;  
end

for i = 1:length(v)-1
    if(expContractVector(i) ~= expContractVector(i+1))
        numberInflexionPoints = numberInflexionPoints+1;
    end
end

freqExpansion = numberInflexionPoints/length(v);
