close all
a=ones(630,457);

img = imread('image.png');
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

r1 = num2str(position(1))
c1 = num2str(position(1)+position(3))
r2 = num2str(position(2))
c2 = num2str(position(2)+position(4))
weak_learner_param = ['HA_Wax' r1 'ay' c1 'bx' r2 'by' c2]
r=mexRectangleFeature(img,weak_learner_param)

weak_learner_param = ['HA_Bax' r1 'ay' c1 'bx' r2 'by' c2]
r=mexRectangleFeature(img,weak_learner_param)