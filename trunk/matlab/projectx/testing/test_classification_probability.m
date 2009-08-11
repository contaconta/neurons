path(path, [pwd '/results/']);  
%load Test_Jul202009-181146calcifer.mat;
load Test_Aug102009calcifer.mat

%I = imread('test_2photon_small.png');
%II = integral_image(I);

I = imread('images/testFIB2.png');
II = integral_image(I);

W = [17 17];
IMSIZE = size(I);

PROBMAP = zeros(size(I));


n = wristwatch('start', 'end', (IMSIZE(1)-W(1))*(IMSIZE(2)-W(2)), 'every', 1000);
count = 0;

for r = 1:IMSIZE(1)-W(1)
    for c = 1:IMSIZE(2)-W(2)
        n = wristwatch(n, 'update', count, 'text', '       classified window ');
	% FIXME : pass image instead of integral image and compute integral image in test_probmap for HA learners
	% That way, we'll be able to handle multiple types of learners
        %PROBMAP(r+floor(W/2), c+floor(W/2)) = test_probmap(CASCADE.CLASSIFIER, II(r:r+W(1)-1,c:c+W(2)-1));
	PROBMAP(r+floor(W/2), c+floor(W/2)) = test_probmap(CASCADE.CLASSIFIER, I(r:r+W(1)-1,c:c+W(2)-1));
        count = count + 1;
    end
end

P_I = mat2gray(PROBMAP);

m = colormap('jet');
imwrite(64*P_I, m, 'result.png', 'PNG');
