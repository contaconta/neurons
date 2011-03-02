

folder = '/home/ksmith/data/Sinergia/Basel/14-11-2010/';

dest = '/home/ksmith/data/Sinergia/Basel/experiments_14-11-2010/';

d = dir(folder);

for k = 3:numel(d)-1
    
    folder_n = [folder d(k).name '/'];
    disp(['WORKING ON ' folder_n]);

    %cmd = ['rm ' folder_n 'output.avi'];
    %disp(cmd);
    %system(cmd);
    
    cmd = ['cp ' folder_n d(k).name '.avi ' dest ];
    disp(cmd);
    system(cmd);
    
    cmd = ['cp ' folder_n d(k).name '.xml ' dest ];
    disp(cmd);
    system(cmd);
    
    
end
