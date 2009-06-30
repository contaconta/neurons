img = imread('image.png');
img = img(:,:,1);
img2 = imread('vesicles.png');
img2 = img2(:,:,1);

learner_param{1} = 'HA_Wax1ay1bx2by2_Bax1ay2bx2by3'
learner_param{2} = 'HA_Wax1ay1bx10by10_Bax1ay10bx10by20'

r=mexRectangleFeature({img,img2},learner_param)
