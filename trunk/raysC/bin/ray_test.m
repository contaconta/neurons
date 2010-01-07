function ray_test(angle)

img_name = {'mitochondria1.png'};
sigma = 20;
rays(img_name,sigma,angle);

figure;
g = imread(img_name{1});
imagesc(g);
axis xy
%print -dpng img2.png

im=read_32bitsimage('ray1.ppm',[size(g,2) size(g,1)]);
imagesc(im);

figure;
im=read_32bitsimage('ray3.ppm',[size(g,2) size(g,1)]);
imagesc(im);

figure;
im=read_32bitsimage('ray4.ppm',[size(g,2) size(g,1)]);
imagesc(im);
