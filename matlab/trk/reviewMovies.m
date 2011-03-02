


folder = '/home/ksmith/data/Sinergia/Basel/14-11-2010/';

d = dir(folder);

reviewlist = [];

count = 1;
for i = 46:140 %[1:20, 121:140]; %67:140
    exp_num(count,:) = sprintf('%03d', i);
    count = count + 1;
end

for k = 1:length(exp_num)

    folder_n = [folder exp_num(k,:) '/'];
    cmd = ['totem ' folder_n  exp_num(k,:) '.avi'];
    disp(cmd);
    system(cmd);
    
    s = input('Accept [space] or review[r]?\n', 's');
    
    if strcmpi(s, 'r')
        reviewlist = [reviewlist; d(k).name];
    end
    
    save([folder 'reviewlist.mat'], 'reviewlist');
    
    disp('');
    disp('=============================================================')
    disp('');
end