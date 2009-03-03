function I = imnormalize(MODE, varargin)
%IMNORMALIZE - normalizes the mean and variance of pixels in an image
%
%   I = imnormalize('image', I);
%   I = imnormalize('integral_images', I);
%
%   REWRITE!!!!
%   I = imnormalize(I, var) takes an input image I and returns an image I 
%   with a uniform intensity variance.  The default variance is 0.055.  As
%   an optional second argument the variance may be specified.
%
%   Copyright © 2008 Kevin Smith
%   See also IMADJUST, IMCONTRAST, INTEGRAL_IMAGE

if ~(strcmp(MODE, 'image') || strcmp(MODE, 'integral_images') || strcmp(MODE, 'integral_images_factors') )
    error('Please specify the MODE as either "image" or "integral_images", or "integral_images_factors".  See "help imnormalize" for details.');
end


% define our target Variance!
DEFAULT_VAR = 0.05;


%% image mode

if strcmp(MODE, 'image')
    I = varargin{1};
    
    % check to see if user specified a desired variance
    if nargin == 3
        VAR = varargin{2};
    else 
        VAR = DEFAULT_VAR;
    end
    
    % check to see if it is class double
    if ~isa(I, 'double')
        cls = class(I);
        I = mat2gray(I, [0 double(intmax(cls))]); 
    end

    % check to see if it is grayscale
    if length(size(I)) ~= 2
        error('imnormalize only works for grayscale images');
    end
    
    % zero the mean, normalize the variance, and re-add the mean
    mu = mean(I(:));
    v = max([0.0000001 var(I(:))]);
    I = (I - mu) / (sqrt(v) / sqrt(VAR) )  + mu;

    % clip the values over 1 and below 0
    I(I > 1) = 1;
    I(I < 0) = 0;
end

%% integral images mode

if strcmp(MODE, 'integral_images')
    I = varargin{1};
    II = varargin{2};  II = II(:);
    II2 = varargin{3}; II2 = II2(:);
    
    % check to see if user specified a desired variance
    if nargin == 5
        VAR = varargin{4};
    else 
        VAR = DEFAULT_VAR;
    end
    
    % check to see if it is class double
    if ~isa(I, 'double')
        cls = class(I);
        I = mat2gray(I, [0 double(intmax(cls))]); 
    end

    % check to see if it is grayscale
    if length(size(I)) ~= 2
        error('imnormalize only works for grayscale images');
    end
    
    % zero the mean, normalize the variance, and re-add the mean
    mu = II(length(II))/length(II);
    v = max([0.0000001     II2(length(II2))/length(II2) - mu^2  ]);
    I = (I - mu) / (sqrt(v) / sqrt(VAR) )  + mu;

    % clip the values over 1 and below 0
    I(I > 1) = 1;
    I(I < 0) = 0;
end

%% integral images factor mode

if strcmp(MODE, 'integral_images_factors')
    
    % check to see if user specified a desired variance
    if nargin == 4
        VAR = varargin{3};
    else 
        VAR = DEFAULT_VAR;
    end
    
    II = varargin{1};  
    II2 = varargin{2}; 
    
    mu = II(length(II))/length(II);
    v = max([0.0000001     II2(length(II2))/length(II2) - mu^2  ]);
    
    % in this case we just return a factor to multiply the feature responses by as I
    % instead of the actual image!
    I = 1 / (sqrt(v) / sqrt(VAR));
end
