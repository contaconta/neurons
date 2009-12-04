%function bgsub(folder)
%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

folder = './cam1/';
outputFolder = './output/';
if ~isdir(outputFolder); mkdir(outputFolder); end;

d = dir([folder '*.jpg']);
if isempty(d)
    d = dir([folder '*.png']);
elseif isempty(d)
    d = dir([folder '*.tif']);
elseif isempty(d)
    d = dir([folder '*.pgm']);
end

%==========================================================================
% PARAMETERS
%==========================================================================
K = 5;                          % the number of gaussians in the model
sigma_init = 40;                % initial variance
%ALPHA = 0.0001; %0.001; %.005;  % adaptation rate
%adapt_speed = .2;    %3;         % speed at which adaptation rate is reached
T =  .35;   %.3   %.45;  %.4         % overall prior probability threshold
lambda = 1.25;  %1.25;       	% distance for a match in STDs
w_init = 0.05;                  % initial prior probability
MINSIZE = 350;   %20;            % size of small objects to remove (in pixels)
%init_imgs = 10;
alpha = .1;
%==========================================================================


%% initialization
I = imread([folder d(1).name]);
H = size(I,1);  W = size(I,2);

w = w_init*ones(H,W,5);
mu = zeros(H,W,3,5);
sigma = zeros(H,W,3,5);
d_h = zeros(H,W,5);
d_s = zeros(H,W,5);
d_v = zeros(H,W,5);
D = zeros(H,W,5);
DMIN = zeros(H,W,5);
MATCHES = zeros(H,W,5);
UNMATCHED = zeros(H,W);
BACKGROUND = zeros(H,W);
LOWEST_RANK_K = zeros(H,W,5);
RANKING = zeros(H,W,5);
labels_hist = zeros(H,W,3);


% set initial values for the gmm matrices
mu(:,:,:,1) = round(255*rand(H,W,3));
sigma(:,:,1:3,1) = sigma_init;

% mu2 = zeros(H,W,3,init_imgs);
% for t = 1:round(length(d)/(init_imgs)):length(d)
%     mu2(:,:,:,t) = double(imread([folder d(t).name]));
% end
% mu(:,:,:,2) = mean(mu2,4);

for k=2:K,
    I = double(imread([folder d(k).name]));
    mu(:,:,:,k) = I(:,:,:);
    sigma(:,:,1:3,k) = sigma_init;
end


Pk = zeros(H,W,5); K_MATCHES_W = Pk;

%% main loop
for t = 1:length(d)
    
    I = imread([folder d(t).name]); Iorig = I; I = double(I);
    
    
    BACKGROUND = zeros(H,W);
    UNMATCHED = zeros(H,W);
    D = zeros(H,W,5); d1=D; d2=D; d3=D;
    K_MATCHES = zeros(H,W,5); 
    LOWEST_RANK_K = zeros(H,W,5);
    RANKING = zeros(H,W,5);
    
    % set the alpha parameter
%     if 1/(alpha_acceleration*t) > ALPHA,
%        alpha = 1/(adapt_speed*t);
%     else
%         alpha = ALPHA;
%     end
    
    disp(['... ' d(t).name ' , alpha = ' num2str(alpha)]);
    

    
    
    % compute the difference matrix
    for k=1:K,
        d1(:,:,k) = (abs(I(:,:,1) - mu(:,:,1,k)))./sigma(:,:,1,k);
        d2(:,:,k) = (abs(I(:,:,2) - mu(:,:,2,k)))./sigma(:,:,2,k);
        d3(:,:,k) = (abs(I(:,:,3) - mu(:,:,3,k)))./sigma(:,:,3,k);
        
        D(:,:,k) = sqrt(d1(:,:,k).^2 + d2(:,:,k).^2 + d3(:,:,k).^2);

        K_MATCHES(:,:,k) = D(:,:,k) <= lambda;
        UNMATCHED = UNMATCHED + K_MATCHES(:,:,k);
    end
    UNMATCHED = UNMATCHED == 0;
    [DMIN] = min(D,[],3);

    for k=1:K,
        K_MATCHES_W(:,:,k) = K_MATCHES(:,:,k).*w(:,:,k);
    end

    [VALS,MATCHES] = max(K_MATCHES_W,[],3);   % pick the highest ranking distribution
    MATCHES = MATCHES.*~UNMATCHED;

    
    for k=1:K,         
        K_MATCHES(:,:,k) = MATCHES == k;

        %  Compute Pk for each gmm
        Pk(:,:,k) = alpha*(K_MATCHES(:,:,k) ./ w(:,:,k) );

        %  Adapt the w for each gaussian
        w(:,:,k) = (1-alpha)*w(:,:,k) + alpha*K_MATCHES(:,:,k);            

        for c = 1:3,
            %  Adapt the mean and variance for each gaussian
            mu(:,:,c,k) = (ones(H,W) - Pk(:,:,k)).*mu(:,:,c,k) + Pk(:,:,k).*I(:,:,c);
            sigma(:,:,c,k) = sqrt( (abs(ones(H,W) - Pk(:,:,k))).*(sigma(:,:,c,k).^2)  +  Pk(:,:,k).*(abs((I(:,:,c) - mu(:,:,c,k)).*2)));
        end
    end
    
        
    for k=1:K,
        w(:,:,k) = w(:,:,k)./sum(w,3);
        BACKGROUND = BACKGROUND + K_MATCHES(:,:,k).*( w(:,:,k) > T);
        RANKING(:,:,k) = (w(:,:,k).^2)./(sqrt(sigma(:,:,1,k).^2 + sigma(:,:,2,k).^2 + sigma(:,:,3,k).^2).^2);
    end

    [VALS, LOWEST_RANK] = min(RANKING,[],3);


    for k=1:K,
        LOWEST_RANK_K(:,:,k) = LOWEST_RANK ==k;
        LOWEST_RANK_K(:,:,k) = LOWEST_RANK_K(:,:,k).*UNMATCHED;

        for c = 1:3,
            mu(:,:,c,k) = mu(:,:,c,k).*(~LOWEST_RANK_K(:,:,k)) + LOWEST_RANK_K(:,:,k).*I(:,:,c);
            sigma(:,:,c,k) = sigma(:,:,c,k).*(~LOWEST_RANK_K(:,:,k)) + LOWEST_RANK_K(:,:,k)*sigma_init;
        end
        w(:,:,k) = w(:,:,k).*(~LOWEST_RANK_K(:,:,k)) + LOWEST_RANK_K(:,:,k)*w_init;
    end
    
    
%     %% morphological operations
%     background = bwmorph(BACKGROUND, 'clean');
% 
%     %se_er = strel('disk',2);    % erode template
%     %se_di = strel('disk',2);    % dilate template
%     
%     FG = ~BACKGROUND;
%     %FG = imdilate(FG,se_di);
%     %FG = imerode(FG,se_er);
%     FG = bwmorph(FG, 'close', 5);
%     FG = bwmorph(FG, 'fill');
%     
%     L = bwlabel(FG,8);    
%     STATS = regionprops(L, 'Area', 'PixelIdxList'); %#ok<MRPBW>
%     for l = 1:length(STATS)
%         if STATS(l).Area < MINSIZE
%             FG(STATS(l).PixelIdxList) = 0;
%         end
%     end
%     
%     BACKGROUND = ~FG;
    
    
    
    I1 = Iorig(:,:,1); I1(BACKGROUND == 1) = 0;
    I2 = Iorig(:,:,2); I2(BACKGROUND == 1) = 0;
    I3 = Iorig(:,:,3); I3(BACKGROUND == 1) = 0;
    Ibg = Iorig;  Ibg(:,:,1) = I1; Ibg(:,:,2) = I2; Ibg(:,:,3) = I3;
    %imshow(imoverlay(Iorig,BACKGROUND, [0 .8 0], 'alpha', 1));
    imshow(Ibg);
    %imagesc(BACKGROUND); 
    axis image; drawnow; refresh; 
    
    filenm = [outputFolder 'frame' number_into_string(t, length(d)) '.png'];
    imwrite(Ibg, filenm, 'PNG');
end


