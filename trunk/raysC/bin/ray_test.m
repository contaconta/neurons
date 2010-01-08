function ray_test(angle)

img_name = {'mitochondria1.png'};
edge_low_threshold = 10000;
edge_high_threshold = 30000;
rays(img_name,angle,edge_low_threshold,edge_high_threshold);

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
