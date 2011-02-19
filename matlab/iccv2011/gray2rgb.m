function  I = gray2rgb(I)
%GRAY2RGB Converts a NxM grayscale image into a NxMx3 color image.
%
%   I = gray2rgb(I) takes a NxM grayscale image as input and coverts it to
%   a NxMx3 RGB color image.  The image itself is not colorized, rather
%   this function makes it possible to create composite images using
%   IMOVERAY, IMLINCOMB, or IMADD.
%
%   Example:
%   ------------
%   I = imread('rice.png');
%   I = gray2rgb(I);
%   I2 = imresize(imread('board.tif'), [size(I,1) size(I,2)]);
%   I3 = imadd(I, I2);
%   figure; imshow(I3);
%   
%   Copyright 2008 Kevin Smith
%
%   See also MAT2GRAY, RGB2GRAY, IND2GRAY, IMOVERLAY, IMADD, IMLINCOMB


I(:,:,2) = I;
I(:,:,3) = I(:,:,1);


