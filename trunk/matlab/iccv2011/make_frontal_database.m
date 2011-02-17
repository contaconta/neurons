

folder = '/home/ksmith/data/faces/wild/train/pos/';

dest = '/home/ksmith/data/faces/frontal/train/pos/';

d = dir([folder '*.png']);
count = 1;

%I = zeros(24,24,length(d));

M = imread('meanface.png');
figure(1); imshow(M); set(gca, 'Position', [0 0 1 1]);

for i = 1:length(d)
    
   
    
    I = imread([folder d(i).name]);
    figure(2); imshow(I); set(gca, 'Position', [0 0 1 1]);
    drawnow;
    r = input(['Keep image ' num2str(i) '?\n'], 's');
    
    if strcmpi(r, 'n')
        
    else
        filename = [dest 'face' sprintf('%05d.png',count)];
        disp(['...writing ' filename]);
        disp(' ');

        imwrite(I, filename, 'PNG');
     	count = count + 1; 
    end
end

