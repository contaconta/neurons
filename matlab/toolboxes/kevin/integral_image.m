function II = integral_image(I)
%INTEGRAL_IMAGE computes the integral image of an input image.
%
%   II = integral_image(I) takes a grayscale image I, computes and returns 
%   the intergral image II. 
%
%   Example:
%   -------------
%   I = imread('circuit.tif');
%   II = integral_image(I);
%   figure; subplot(1,2,1); imshow(I); subplot(1,2,2); imagesc(II); 
%   axis image;
%
%   Copyright 2008 Kevin Smith
%
%   See also IMREAD, VJ_TRAIN

if (strncmp(class(I), 'uint', 4)) 
    I = double(I);
    II = cumsum(cumsum(I),2);
    II = uint32(II);
else
    II = cumsum(cumsum(I),2);
end