img = imread('Images/img1.jpg');
img = img(:,:,1);

%figure, imshow(img);
%h = imrect;
%position = wait(h)
position = [136.0000  135.0000  125.0000  105.0000];
% position has the form [xmin ymin width height]
masked_img=img(position(2):position(2)+position(4), position(1):position(1)+position(3));
%imshow(masked_img);

display 'HA_Wax0ay0bx10by10_Bax0ay10bx10by20'
r=mexRectangleFeature(img,'HA_Wax0ay0bx10by10_Bax0ay10bx10by20')
