close all
a=ones(630,457);

img = imread('Images/img1.jpg');
img = img(:,:,1);

int_img=mexIntegralImage(img,a);
size(int_img)
imagesc(int_img)
%imshow(uint8(int_img*(255.0/max(max(int_img)))));

figure, imshow(img);
h = imrect;
position = wait(h)
% position has the form [xmin ymin width height]
masked_img=img(position(2):position(2)+position(4), position(1):position(1)+position(3));
imshow(masked_img);

%row, col, rows, cols
r=mexBoxIntegral(img,position)

r=mexRectangleFeature(img,position,0)
