
matlabpool

%% ICFILER
%folder = '/net/cvlabfiler1/home/ksmith/Basel/14-11-2010/';
%resultsFolder = '/net/cvlabfiler1/home/ksmith/Basel/Results/';

%% kevin's laptop
folder = '/home/ksmith/data/Sinergia/Basel/14-11-2010/';
resultsFolder = '/home/ksmith/data/Sinergia/Basel/Results/';



d = dir(folder);

addpath('code/');

count = 1;
for i = 4:4
    exp_num(count,:) = sprintf('%03d', i); %#ok<SAGROW>
    count = count + 1;
end

for i = 1:size(exp_num,1)
    tic
    folder_n = [folder exp_num(i,:) '/'];
    trkTracking(folder_n, resultsFolder);
    disp('');
    disp('=============================================================')
    disp('');
    toc
end

% kill the matlab pool
matlabpool close force