function [MASK] = vj_plot_haar_feature(varargin)
%VJ_PLOT_HAAR_FEATURE plots haar-like weak classifiers
%
%   [MASK] = vj_plot_haar_feature(FEATURE, polarity, IMSIZE, I, D) plots a haar-like 
%   feature vector defined by FEATURE for visualization onto a region 
%   defined by IMSIZE (the standard training image size). Optional argument
%   'image' followed by image I overlays the feature onto I. Optional 
%   argument 'D" followed by a booloean toggles plot on/off (set to 0 if 
%   you want the mask returned without plotting).  Returns the image MASK 
%   defining the haar Feature.
%
%   Example 1:
%   ------------------
%   IMSIZE = [24 24];                                   % training image size        
%   F = [1 11 9 14 17 15 9 18 17 0 0 0 0 0 0 0 0];      % haar-like feature
%   vj_plot_haar_feature(F, 1, IMSIZE);
%   
%   Example 2:
%   ------------------
%   IMSIZE = [24 24];                                   % training image size        
%   F = [1 11 9 14 17 15 9 18 17 0 0 0 0 0 0 0 0];      % haar-like feature
%   M = vj_plot_haar_feature(F, 1, IMSIZE, 'D', 0);
%   figure; imagesc(M);  axis square; colormap(gray);
%
%   Example 3:
%   ------------------
%   IMSIZE = [24 24];                                   % training image size        
%   F = [1 11 9 14 17 15 9 18 17 0 0 0 0 0 0 0 0];      % haar-like feature
%   vj_plot_haar_feature(F, 1, IMSIZE, 'image', mat2gray(imread('circuit.tif')));
%
%   Copyright 2008 Kevin Smith
%
%   See also IMREAD, VJ_TRAIN, VJ_DEFINE_WEAK_CLASSIFIERS

% define parameters
FEATURE = varargin{1}; polarity = varargin{2};
IMSIZE = varargin{3};IMAGE_WIDTH = IMSIZE(1);  IMAGE_HEIGHT = IMSIZE(2);
MASK = zeros([IMAGE_WIDTH IMAGE_HEIGHT]);  D = 1;

for i = 3:nargin
    if strcmp(varargin{i}, 'image')
        I = varargin{i+1};
        dim = size(I);
        if ~isequal([dim(1) dim(2)], IMSIZE)
            I = imresize(I, IMSIZE);
        end
    end
    if strcmp(varargin{i}, 'D')
        D = varargin{i+1};
    end
end
   

% create a haar-like feature mask from FEATURE, depending on the feature type
switch FEATURE(1) 
    case 1
        W1x1 = FEATURE(2);
        W1y1 = FEATURE(3);
        W1x2 = FEATURE(4);
        W1y2 = FEATURE(5);
        B1x1 = FEATURE(6); 
        B1y1 = FEATURE(7);
        B1x2 = FEATURE(8);
        B1y2 = FEATURE(9);
        
%         MASK(W1y1:W1y2,W1x1:W1x2) = polarity*1;
%         MASK(B1y1:B1y2,B1x1:B1x2) = polarity*-1;
        MASK(W1y1:W1y2,W1x1:W1x2) = MASK(W1y1:W1y2,W1x1:W1x2) + ones(size(MASK(W1y1:W1y2,W1x1:W1x2)))*polarity;
        MASK(B1y1:B1y2,B1x1:B1x2) = MASK(B1y1:B1y2,B1x1:B1x2) - ones(size(MASK(W1y1:W1y2,W1x1:W1x2)))*polarity;
        
        
        
        
        %MASK(W1x1:W1x2, W1y1:W1y2) = polarity*1;
        %MASK(B1x1:B1x2, B1y1:B1y2) = polarity*-1;
    case 2
        W1x1 = FEATURE(2);
        W1y1 = FEATURE(3);
        W1x2 = FEATURE(4);
        W1y2 = FEATURE(5);
        B1x1 = FEATURE(6); 
        B1y1 = FEATURE(7);
        B1x2 = FEATURE(8);
        B1y2 = FEATURE(9);
        
        MASK(W1y1:W1y2,W1x1:W1x2) = MASK(W1y1:W1y2,W1x1:W1x2)+ ones(size(MASK(W1y1:W1y2,W1x1:W1x2)))*polarity;
        MASK(B1y1:B1y2,B1x1:B1x2) = MASK(B1y1:B1y2,B1x1:B1x2) - ones(size(MASK(B1y1:B1y2,B1x1:B1x2)))*polarity;
%         MASK(W1y1:W1y2,W1x1:W1x2) = polarity*1;
%         MASK(B1y1:B1y2,B1x1:B1x2) = polarity*-1;
    case 3
        W1x1 = FEATURE(2);
        W1y1 = FEATURE(3);
        W1x2 = FEATURE(4);
        W1y2 = FEATURE(5);
        B1x1 = FEATURE(6); 
        B1y1 = FEATURE(7);
        B1x2 = FEATURE(8);
        B1y2 = FEATURE(9);
        
        MASK(W1y1:W1y2,W1x1:W1x2) = MASK(W1y1:W1y2,W1x1:W1x2) + ones(size(MASK(W1y1:W1y2,W1x1:W1x2))) * polarity;
        MASK(B1y1:B1y2,B1x1:B1x2) = MASK(B1y1:B1y2,B1x1:B1x2) - ones(size(MASK(B1y1:B1y2,B1x1:B1x2))) * polarity;
%         MASK(W1y1:W1y2,W1x1:W1x2) = polarity*1;
%         MASK(B1y1:B1y2,B1x1:B1x2) = polarity*-1;
        
        if FEATURE(10) ~= 0
            W2x1 = FEATURE(10);
            W2y1 = FEATURE(11);
            W2x2 = FEATURE(12);
            W2y2 = FEATURE(13);
            MASK(W2y1:W2y2,W2x1:W2x2) = MASK(W2y1:W2y2,W2x1:W2x2)+ ones(size(MASK(W2y1:W2y2,W2x1:W2x2)))*polarity;
            %MASK(W2y1:W2y2,W2x1:W2x2) = polarity*1;
        else
            B2x1 = FEATURE(14); 
            B2y1 = FEATURE(15);
            B2x2 = FEATURE(16);
            B2y2 = FEATURE(17);
            MASK(B2y1:B2y2,B2x1:B2x2) = MASK(B2y1:B2y2,B2x1:B2x2)-ones(size(MASK(B2y1:B2y2,B2x1:B2x2)))*polarity;
            %MASK(B2y1:B2y2,B2x1:B2x2) = polarity*-1;
        end
        
    case 4
        W1x1 = FEATURE(2);
        W1y1 = FEATURE(3);
        W1x2 = FEATURE(4);
        W1y2 = FEATURE(5);
        B1x1 = FEATURE(6); 
        B1y1 = FEATURE(7);
        B1x2 = FEATURE(8);
        B1y2 = FEATURE(9);
        
        MASK(W1y1:W1y2,W1x1:W1x2) = MASK(W1y1:W1y2,W1x1:W1x2)+ones(size(MASK(W1y1:W1y2,W1x1:W1x2)))* polarity;
        MASK(B1y1:B1y2,B1x1:B1x2) = MASK(B1y1:B1y2,B1x1:B1x2)-ones(size(MASK(B1y1:B1y2,B1x1:B1x2)))*polarity;
%         MASK(W1y1:W1y2,W1x1:W1x2) = polarity*1;
%         MASK(B1y1:B1y2,B1x1:B1x2) = polarity*-1;
        
        if FEATURE(10) ~= 0
            W2x1 = FEATURE(10);
            W2y1 = FEATURE(11);
            W2x2 = FEATURE(12);
            W2y2 = FEATURE(13);
            MASK(W2y1:W2y2,W2x1:W2x2) = MASK(W2y1:W2y2,W2x1:W2x2)+ones(size(MASK(W2y1:W2y2,W2x1:W2x2)))*polarity;
            %MASK(W2y1:W2y2,W2x1:W2x2) = polarity*1;
        else
            B2x1 = FEATURE(14); 
            B2y1 = FEATURE(15);
            B2x2 = FEATURE(16);
            B2y2 = FEATURE(17);
            MASK(B2y1:B2y2,B2x1:B2x2) = MASK(B2y1:B2y2,B2x1:B2x2)-ones(size(MASK(B2y1:B2y2,B2x1:B2x2)))*polarity;
            %MASK(B2y1:B2y2,B2x1:B2x2) = polarity*-1;
        end    
        
    case 5
        W1x1 = FEATURE(2);
        W1y1 = FEATURE(3);
        W1x2 = FEATURE(4);
        W1y2 = FEATURE(5);
        B1x1 = FEATURE(6); 
        B1y1 = FEATURE(7);
        B1x2 = FEATURE(8);
        B1y2 = FEATURE(9);
        W2x1 = FEATURE(10);
        W2y1 = FEATURE(11);
        W2x2 = FEATURE(12);
        W2y2 = FEATURE(13);
        B2x1 = FEATURE(14);
        B2y1 = FEATURE(15);
        B2x2 = FEATURE(16);
        B2y2 = FEATURE(17);
        
        MASK(W1y1:W1y2,W1x1:W1x2) = MASK(W1y1:W1y2,W1x1:W1x2)+ones(size(MASK(W1y1:W1y2,W1x1:W1x2)))*polarity;
        MASK(B1y1:B1y2,B1x1:B1x2) = MASK(B1y1:B1y2,B1x1:B1x2)-ones(size(MASK(B1y1:B1y2,B1x1:B1x2)))*polarity;
        MASK(W2y1:W2y2,W2x1:W2x2) = MASK(W2y1:W2y2,W2x1:W2x2)+ones(size(MASK(W2y1:W2y2,W2x1:W2x2)))*polarity;
        MASK(B2y1:B2y2,B2x1:B2x2) = MASK(B2y1:B2y2,B2x1:B2x2)-ones(size(MASK(B2y1:B2y2,B2x1:B2x2)))*polarity;
%         MASK(W1y1:W1y2,W1x1:W1x2) = polarity*1;
%         MASK(B1y1:B1y2,B1x1:B1x2) = polarity*-1;
%         MASK(W2y1:W2y2,W2x1:W2x2) = polarity*1;
%         MASK(B2y1:B2y2,B2x1:B2x2) = polarity*-1;
     
    otherwise
        disp('ERROR: unexpected feature type');
        keyboard;
end

if D
    % Plot the haar-like feature
    %figure(12210); colormap(gray);
    figure;  colormap(gray);
    if ~exist('I', 'var');
        imagesc(MASK); 
    else
        imshow(imadd(MASK,I));
    end
    line( [W1x1-.5 W1x2+.5], [W1y1-.5 W1y1-.5], 'Color', [0 0 0]);
    line( [W1x1-.5 W1x2+.5], [W1y2+.5 W1y2+.5], 'Color', [0 0 0]);
    line( [W1x1-.5 W1x1-.5], [W1y1-.5 W1y2+.5], 'Color', [0 0 0]);
    line( [W1x2+.5 W1x2+.5], [W1y1-.5 W1y2+.5], 'Color', [0 0 0]);
    
    line( [B1x1-.5 B1x2+.5], [B1y1-.5 B1y1-.5], 'Color', [0 0 0]);
    line( [B1x1-.5 B1x2+.5], [B1y2+.5 B1y2+.5], 'Color', [0 0 0]);
    line( [B1x1-.5 B1x1-.5], [B1y1-.5 B1y2+.5], 'Color', [0 0 0]);
    line( [B1x2+.5 B1x2+.5], [B1y1-.5 B1y2+.5], 'Color', [0 0 0]);
    axis square;
    refresh;
end