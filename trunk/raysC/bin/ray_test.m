function ray_test(angle)

img_name = {'mitochondria2.png'};
rays(img_name,20,angle);

figure;
g = imread('img.png');
imagesc(g);
axis xy
print -dpng img2.png

im = imread('ray1.png');
imagesc(im);

figure;
im = imread('ray3.png');
imagesc(im);

figure;
im = imread('ray4.png');
imagesc(im);
