


folder = '/home/ksmith/data/Sinergia/Basel/14-11-2010/';

d = dir(folder);

count = 1;
for i = 83:140
    exp_num(count,:) = sprintf('%03d', i); %#ok<SAGROW>
    count = count + 1;
end

for i = 1:length(exp_num)
    
    folder_n = [folder exp_num(i,:) '/'];
    trkTracking(folder_n);
    disp('');
    disp('=============================================================')
    disp('');
end