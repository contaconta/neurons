

%I0 = imread('cameraman.tif');
I0 = imread('/osshare/Work/Data/LabelMe/Images/FIBSEM/FIBSLICE0002.png');
I0 = impyramid(I0, 'reduce');
tic;

WSIZE = 10;
r = 10*rand(2*WSIZE + 1);
r = uint8(r);


for i = WSIZE+1:size(I0,1)-WSIZE-1
    for j = WSIZE+1:size(I0,2) - WSIZE-1
        wind = I0(i-WSIZE:i+WSIZE,j-WSIZE:j+WSIZE);
        %wind = wind + r + r + reshape(wind(1:numel(wind)), size(wind,1),size(wind,2));
        %w = imrotate(wind,-20, 'crop');
        w = fast_rotate(wind,-20);
        %wind = imrotate(I0(i-5:i+5,j-5:j+5), -20, 'bilinear', 'crop');
    end
end

toc;