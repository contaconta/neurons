img = imread('Images/img1.jpg');
img = img(:,:,1);

%figure, imshow(img);
%h = imrect;
%position = wait(h)
position = [136.0000  135.0000  125.0000  105.0000];
% position has the form [xmin ymin width height]
masked_img=img(position(2):position(2)+position(4), position(1):position(1)+position(3));
%imshow(masked_img);

learner_param{1} = 'HA_Wax1ay1bx2by2_Bax1ay2bx2by3'
learner_param{2} = 'HA_Wax1ay1bx10by10_Bax1ay10bx10by20'
r=mexRectangleFeature(img,learner_param)
