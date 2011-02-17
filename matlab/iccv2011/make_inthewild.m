folder = '/home/ksmith/data/faces/frontal/train/pos/';

dest = '/home/ksmith/data/faces/inthewild/train/pos/';

d = dir([folder '*.png']);
count = 1;

pat = '\d*';

count = 1;

for i = 1:length(d)

    
    fname = [folder d(i).name];
    I = imread([folder d(i).name]);
    %figure(2); imshow(I); set(gca, 'Position', [0 0 1 1]);
    %drawnow;
    
    num = regexp(fname, pat, 'match');
    
    num = str2double(num{1});
    
    
    if num >= 3235
    
       	filename = [dest 'face' sprintf('%05d.png',count)];
        disp(['...writing ' filename]);
        disp(' ');

        imwrite(I, filename, 'PNG');
     	count = count + 1; 
        
        
    end
end
    
    