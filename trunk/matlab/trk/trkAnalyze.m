
matlabpool

folder = '/media/basel/Basel/14-11-2010/';

d = dir(folder);

addPath('code/');

count = 1;
for i = 1:1
    exp_num(count,:) = sprintf('%03d', i); %#ok<SAGROW>
    count = count + 1;
end

for i = 1:size(exp_num,1)
    tic
    folder_n = [folder exp_num(i,:) '/'];
    trkTracking(folder_n);
    disp('');
    disp('=============================================================')
    disp('');
    toc
end