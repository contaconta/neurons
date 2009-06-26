img = imread('image.png');
img = img(:,:,1);

%figure, imshow(img);
%h = imrect;
%position = wait(h)
position = [136.0000  135.0000  125.0000  105.0000];
% position has the form [xmin ymin width height]
%masked_img=img(position(2):position(2)+position(4), position(1):position(1)+position(3));
%imshow(masked_img);

learner_param{1} = 'HA_Wax0ay0bx1by1_Bax0ay1bx1by2'
learner_param{2} = 'HA_Wax0ay0bx9by9_Bax0ay9bx9by19'
r=mexRectangleFeature(img,learner_param)
