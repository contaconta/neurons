function II = integralImage(I,type)
%INTEGRALIMAGE computes the integral image of an input image.
%
%   II = integralImage(I) computes the integral image or summed-area-table
%   of a grayscale image I. II = integralImage(I, 'x') computes a
%   semi-summed area table in direction x, while option 'y' computes a
%   semi-summed area table in direction y.
%
%   Example:
%   -------------
%   I = imread('circuit.tif');
%   II = integral_image(I);
%   IIx = integralImage(I,'x');
%   figure; subplot(1,3,1); imshow(I); subplot(1,3,2); imagesc(II);
%   subplot(1,3,3); imagesc(IIx); axis image;
%
%   Copyright 2010 Kevin Smith
%
%   See also IMREAD

%% check to see if the original image I is a uint, convert to double in
%% this case
wasuint = 0;
if (strncmp(class(I), 'uint', 4))
    wasuint = 1;
    I = double(I);
end


%% compute the integral image as either 1) a normal summed area table, 2) a
%% semi-summed table along the y direction, 3) a semi-summed table along
%% the x direction
if ~exist('type', 'var')
    II = cumsum(cumsum(I,2));
    maxval = II(size(II,1),size(II,2));
elseif strcmp(type, 'outer')
    II = cumsum(cumsum(I,2));
    II = padarray(II, [1 1], 0, 'pre');
%     Itemp = cumsum(I,2)-I;
%     Itemp(2:size(I,1),1) = I(1:size(I,1)-1,1);
%     II = cumsum(Itemp,1)-Itemp;
    maxval = II(size(II,1), size(II,2));
    
elseif strcmp(type, 'y')
    II = cumsum(I,1);
    maxval = max(II(size(II,1), :));
elseif strcmp(type, 'x')
    II = cumsum(I,2);
    maxval = max(II(:, size(II,2)));
else
    error('Error: summed area type must be specified as either "x" or "y".');
end

%% if the original image was a uint, convert back to a uint of appropriate
%% size
if wasuint  
    if maxval < intmax('uint16')
        II = uint16(II);
    elseif maxval < intmax('uint32')
        II = uint32(II);
    elseif maxval < intmax('uint64')
        II = uint64(II);
    else
        error('Error - image is too large to represent with uint64.');
    end
end