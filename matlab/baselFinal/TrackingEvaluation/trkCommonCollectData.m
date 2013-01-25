function [X, S, D, posit, negat, both] = trkCommonCollectData(indices, WIND_SIZE, Frames, isWithRandom)

X = [];
for j = indices
    curFrame = Frames(j);
    for i = 1:size(curFrame.CommonCells, 2)
        char = [...
  j,... % 1
  i,... %2
  curFrame.CommonCells(i).Time,... % 3
  curFrame.CommonCells(i).SomaGreenHistogram(1:end)',...%4..23
  curFrame.CommonCells(i).SomaCentroid,...%24..25
  curFrame.CommonCells(i).SomaArea,...%26
  curFrame.CommonCells(i).SomaEccentricity,...%27
  curFrame.CommonCells(i).SomaPerimeter,...%28
  curFrame.CommonCells(i).SomaCircularity,...%29
  curFrame.CommonCells(i).SomaOrientation,...%30
  curFrame.CommonCells(i).SomaMeanGreenIntensity,...%31
                ];
        X = [X; char];
    end
end
numOfExamples = size(X,1);
S = zeros(numOfExamples,numOfExamples);
D = zeros(numOfExamples,numOfExamples);
posit = [];
negat = [];
both = [];
NUMBER_OF_BINS = Frames(indices(1)).NUMBER_OF_BINS;
punishMatrix = ones(NUMBER_OF_BINS, NUMBER_OF_BINS);
for i = 1:NUMBER_OF_BINS
    for j = 1:NUMBER_OF_BINS
        punishMatrix(i,j) = abs(i-j);
    end
end

begs = ones(1, max(indices))*numOfExamples;
ends = zeros(1, max(indices));
for i = 1:numOfExamples
    ind = X(i,1);
    begs(ind) = min(begs(ind), i);
    ends(ind) = i;
end

numS = 0;
numD = 0;
for ij = indices
    for i = begs(ij):ends(ij)
        for j = i+1:ends(ij)
            if (abs(X(i,3)-X(j,3)) < WIND_SIZE)
                continue;
            end
            if (abs(X(i,3)-X(j,3)) == WIND_SIZE)
                eqID = (Frames(X(i,1)).CommonCells(X(i,2)).ID == Frames(X(j,1)).CommonCells(X(j,2)).ID);
                if isWithRandom && (~eqID) && (rand < (numD+1)/(numS+1)*0.666)
                    continue;
%                 elseif (eqID) && (abs(X(i,3)-X(j,3)) ~= WIND_SIZE)
%                     continue;
                end
                cell1 = Frames(X(i,1)).CommonCells(X(i,2));
                cell2 = Frames(X(j,1)).CommonCells(X(j,2));
                P = cell1.SomaGreenHistogram;
                Q = cell2.SomaGreenHistogram;
                somaDist = emd_hat_gd_metric_mex(P,Q,punishMatrix,-1);
                P = cell1.SomaGreenHistogramNormalized;
                Q = cell2.SomaGreenHistogramNormalized;
                somaDistNormalized = emd_hat_gd_metric_mex(P,Q,punishMatrix,-1);
                centroidsDist = trkCommonDist(cell1.SomaCentroid, cell2.SomaCentroid);
% %                 if (sum(P.*Q) < 100)
% %                     centroidsDist = 1000;
% %                 end
                P = cell1.NucleusGreenHistogram;
                Q = cell2.NucleusGreenHistogram;
                nucleusDist = emd_hat_gd_metric_mex(P,Q,punishMatrix,-1);
                P = cell1.NucleusGreenHistogramNormalized;
                Q = cell2.NucleusGreenHistogramNormalized;
                nucleusDistNormalized = emd_hat_gd_metric_mex(P,Q,punishMatrix,-1);
                
                dists = [i, j, somaDist, somaDistNormalized,...
                               nucleusDist, nucleusDistNormalized,...
                               centroidsDist, 0];
                if (eqID)
                    S(i,j) = 1;
                    posit = [posit; dists];
                    numS = numS + 1;
                else
                    numD = numD + 1;
                    D(i,j) = 1;
                    negat = [negat; dists];
                end
                both = [both; dists];
            else
                break;
            end
        end
    end
end