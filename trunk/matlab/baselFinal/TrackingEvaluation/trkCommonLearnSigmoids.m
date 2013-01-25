%%
disp('...loading Frames');
clear all
load Frames6.mat
numOfFrames = size(Frames, 2);
numOfHalf = floor(numOfFrames/2);
trainIndices = 1:numOfHalf;
testIndices = numOfHalf+1:numOfFrames;
%%
tic
disp('...gathering data for metrics');
WIND_SIZE = 20;
tic
[trainX, trainS, trainD, trainPosit, trainNegat, ~] = trkCommonCollectData(trainIndices, WIND_SIZE, Frames, true);
toc
[sum(sum(trainS)), sum(sum(trainD))]
%%
tic
disp('...learning sigmas');
sigmas = [];
% characteristics:
% 1 - somaDist, 
% 2 - somaDistNormalized, 
% 3 - nucleusDist, 
% 4 - nucleusDistNormalized, 
% 5 - centroidsDist, 
% 6 - Xing/Ng,
% 7 - lmnn2,
% 8 - importance weights
for curChar = 1:5
    x = [trainPosit(:,3:end); trainNegat(:,3:end)]; x = x(:, curChar);
    y = [ones(1,size(trainPosit,1)),zeros(1,size(trainNegat,1))]';
    [b,dev,stats] = glmfit(x,y,'binomial','link','logit');
    sigmas = [sigmas, b];
end