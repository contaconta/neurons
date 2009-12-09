function ray_test(angle)

img_name = {'mitochondria2.png'};
rays(img_name,20,angle);

im = imread('ray1.png');
figure;
imagesc(im);
