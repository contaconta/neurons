function [FP, TP, FN, TN] = getROCRates(PositiveEMDs, NegativeEMDs, NbPoints)

minValue = min(min(PositiveEMDs), min(NegativeEMDs));
maxValue = max(max(PositiveEMDs), max(NegativeEMDs));

FP = zeros(NbPoints, 1);
TP = zeros(NbPoints, 1);
FN = zeros(NbPoints, 1);
TN = zeros(NbPoints, 1);


idx = 1;
for t = linspace(minValue, maxValue, NbPoints)
   FP(idx) = sum(NegativeEMDs <  t) / numel(NegativeEMDs);
   TN(idx) = sum(NegativeEMDs >= t) / numel(NegativeEMDs);
   FN(idx) = sum(PositiveEMDs >= t) / numel(PositiveEMDs);
   TP(idx) = sum(PositiveEMDs <  t) / numel(PositiveEMDs);
   idx = idx + 1;
end
