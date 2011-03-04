%% Loads the experiment
close all;
clear all;

addpath('~/opt.source/FastEMD/');

TRIALS(1) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '14-11-2010_%03i.mat');
TRIALS(2) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '15-11-2010_%03i.mat');
TRIALS(3) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '16-11-2010_%03i.mat');
TRIALS(4) = LoadTrial('/net/cvlabfiler1/home/ksmith/Basel/Results/', '17-11-2010_%03i.mat');


%% Computes the values to compare into
min_val = 0;
max_val = 0;
funcHandle = @GetNeuriteLengthFromExperiment;
%funcHandle = @GetNumberOfBranchingPointsPerNeurite;




% Gets the responses for the function for each experiment
for nE = 1:length(EXP)
   resp{nE} = funcHandle(EXP{nE}, min_val); 
   maxv = max(resp{nE});
   minv = min(resp{nE});
   if( maxv > max_val)
       max_val = maxv;
   end
   if( minv < min_val)
       min_val = minv;
   end
end


%% Builds the histograms
nBins = 50;
X = min_val:(max_val - min_val)/nBins:max_val;
for nE = 1:1:length(EXP)
   n{nE} = hist(resp{nE}, X);
   n{nE} = n{nE} ./ sum(n{nE});
end


%% Compares the histograms

% Builds the parameters for the Earth Mover's Distance
N= length(X);
THRESHOLD= 3;
extra_mass_penalty= -1;
flowType= 1;
D= ones(N,N).*THRESHOLD;
for i=1:N
    for j=max([1 i-THRESHOLD+1]):min([N i+THRESHOLD-1])
        D(i,j)= abs(i-j); 
    end
end


%% Finally computes the matrix;
Similarity = zeros(length(resp));
for i = 1:1:length(resp)
    for j = i:1:length(resp)
        similarity(i,j) = emd_hat_gd_metric_mex(n{i}', n{j}', D, extra_mass_penalty, flowType);
        similarity(j,i) = similarity(i,j);
    end
end

%%
figure;
imagesc(similarity)



%% Tests the contents of the experiment
% figure;
% for t = 1:1:length(EXP{1}.Dlist)
%     renderData(EXP{1}, t);
%     pause(0.1);
% end
% 
% 
% %% Follows the life of a neuron
% renderNeuronTrack(EXP{1}, 12)
