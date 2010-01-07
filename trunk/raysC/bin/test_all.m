img_name = {'mitochondria2.png'};

g=imread(img_name{1});

for angle=0:1:360
disp(['angle ' int2str(angle) '!!!'])

rays(img_name,20,angle);

figure(1);
im=read_32bitsimage('ray1.ppm',[size(g,2) size(g,1)]);
imagesc(im);
refresh;
pause(0.1);

figure(2);
im=read_32bitsimage('ray3.ppm',[size(g,2) size(g,1)]);
imagesc(im);
refresh;
pause(0.1);

figure(3);
im=read_32bitsimage('ray4.ppm',[size(g,2) size(g,1)]);
imagesc(im);
refresh;
pause(0.1);

end
