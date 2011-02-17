
folder1 = '/home/ksmith/data/faces/EPFL-CVLAB_faceDB/train/pos/';
folder2 = '/home/ksmith/Downloads/lfwcrop_grey/faces/';

dest = '/home/ksmith/data/faces/wild/train/pos/';

d1 = dir([folder1 '*.png']);
d2 = dir([folder2 '*.pgm']);
count = 1;

for i = 1:length(d1)
    
    str = ['cp ' folder1 d1(i).name ' ' dest 'face' sprintf('%05d.png',i)];
    disp(str);
    system(str);
    count = count + 1;
end


for i = 1:length(d2);
    
    I = imread([folder2 d2(i).name]);
    I = imresize(I, [24 24]);
    
    filename = [dest 'face' sprintf('%05d.png',count)];
    disp(['writing ' filename]);

    imwrite(I, filename, 'PNG');
    count = count + 1; 
end