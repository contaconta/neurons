function [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(v)

PERCENT_CHANGE_EXPAND = 0.05;

expContractVector = zeros(size(v));

numberInflexionPoints = 0;
timeExpanding = 0;
timeContracting = 0;

for i = 1:length(v)-1
    if abs(  (v(i+1) - v(i) ) / v(i) ) > PERCENT_CHANGE_EXPAND
    %if abs( v(i+1) - v(i) ) > 6 
    
        if(v(i+1) > v(i))
            expContractVector(i+1) = 1;
            timeExpanding = timeExpanding + 1;
        else
            expContractVector(i+1) = -1; 
            timeContracting = timeContracting + 1;  
        end
    else
        expContractVector(i+1) = 0;
    end  
end

% for i = 1:length(v)-1
%     if(v(i+1) > v(i))
%         expContractVector(i) = 1;
%         timeExpanding = timeExpanding + 1;
%     else
%         expContractVector(i) = -1; 
%         timeContracting = timeContracting + 1;  
%     end
% end


expContractVector(1) = expContractVector(2);
if(expContractVector(1) == 1)
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
