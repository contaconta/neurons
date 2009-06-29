

img = imread('image.png');

NSAMPLES = 1000;
IMSIZE = [23 23];

for n = 1:NSAMPLES
    Wax = ceil(IMSIZE(2)*rand([1 1]));
    Way = ceil(IMSIZE(1)*rand([1 1]));
    Wbx = ceil(Wax+(IMSIZE(2)+1-Wax)*rand([1 1]));
    Wby = ceil(Way+(IMSIZE(1)+1-Way)*rand([1 1]));

    Bax = ceil(IMSIZE(2)*rand([1 1]));
    Bay = ceil(IMSIZE(1)*rand([1 1]));
    Bbx = ceil(Bax+(IMSIZE(2)+1-Bax)*rand([1 1]));
    Bby = ceil(Bay+(IMSIZE(1)+1-Bay)*rand([1 1]));

    learner = ['HA_' 'Wax' num2str(Wax) 'ay' num2str(Way) 'bx' num2str(Wbx) 'by' num2str(Wby) '_Bax' num2str(Bax) 'ay' num2str(Bay) 'bx' num2str(Bbx) 'by' num2str(Bby)];
    disp(learner);
    visualize_haar_feature({learner}, [24 24], mat2gray(img))
    
    
    
    rmex = mexRectangleFeature(img, {learner});
    
    rmat = sum(sum(img(Way:Wby-1,Wax:Wbx-1))) - sum(sum(img(Bay:Bby-1,Bax:Bbx-1)));
    disp('-----------------------------------');
    disp(['rmex: ' num2str(rmex) ' , rmat: ' num2str(rmat) ]);
    
    if rmat ~= rmex
        disp('Error!  MEX file did not match Matlab calculation!');
        keyboard;
    end
end