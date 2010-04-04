function gaussianpyramid(I)


S = 3;
sigma0 = .5;  %.5

I = double(I); I = impyramid(I, 'expand');

% THIS IS IMPORTANT!! COSTS MORE WHEN LARGE
%max_sigma = min(size(I))/(6);
max_sigma = min(size(I));

k = 2^(1/S);
sigma_list  = get_sigmas(sigma0, max_sigma,k);

%I = double(I); I = impyramid(I, 'expand');

G = zeros(size(I,1), size(I,2), length(sigma_list));
DoG = zeros(size(I,1), size(I,2), length(sigma_list)-1 );

tic;
s = 1;
for sigma_i = sigma_list
    disp(['sigma_i = ' num2str(sigma_i)]);
    
%    G(:,:,s) = imgaussian(I, sigma_i);
    %figure; imshow(uint8(G(:,:,s)));
    
    
    hsize = round([6*sigma_i+1, 6*sigma_i+1]);   % The filter size.
    gaussian = fspecial('gaussian',hsize,sigma_i);
%    G(:,:,s) = filter2(gaussian,I);        % Smoothed image.
     G(:,:,s) = imfilter(I,gaussian, 'replicate');        % Smoothed image.
    
    if s > 1
        %figure; imshow(uint8( G(:,:,s) - G(:,:,s-1) ));
        %DoG(:,:,s-1) = abs(G(:,:,s) - G(:,:,s-1));
        DoG(:,:,s-1) = abs(G(:,:,s) - G(:,:,s-1));
        %figure; imagesc( DoG(:,:,s-1) ); axis image; colormap gray;
    end
    
    s = s + 1;
end
toc;
    
[M scale_estimate] = max(DoG, [], 3);

% % for s = 1:length(sigma_list)
% %     scale_estimate(scale_estimate == s) = sigma_list(s);
% % end
 figure; imagesc(scale_estimate); axis image; colormap gray;
% 
% R = '';
% while strcmp(R, '')
% 
%     r = randsample(numel(scale_estimate),1);
%     P = zeros(1,2);
%     [P(1) P(2)] = ind2sub(size(scale_estimate), r);
%     plotcircle(20, 3*scale_estimate(P(1), P(2)), P);
%     R = input('Press any key to show another point, type something to quit.\n', 's');
% end


% I0 = I;
% %I0 = imread('cameraman.tif');
% I1 = impyramid(I0, 'reduce');
% I2 = impyramid(I1, 'reduce');
% I3 = impyramid(I2, 'reduce');
% imshow(I0)
% figure, imshow(I1)
% figure, imshow(I2)
% figure, imshow(I3)

keyboard;


function sigma_list  = get_sigmas(sigma_list, max_sigma,k)

if sigma_list(length(sigma_list))*k < max_sigma
    sigma_list = [sigma_list sigma_list(length(sigma_list))*k];
    %keyboard;
    sigma_list = get_sigmas(sigma_list, max_sigma, k);
end



function plotcircle(n, R, C)
% PlotCircle_0.m:   Draw a circle of radius 1 centered at the origin
%    n = # points
%    r = radius
%    c = center
%    Drawing is a set of lines between points: the more points, the
%    better approximation to a circle.
% Input: number of points

angle = 0:2*pi/n:2*pi;            % vector of angles at which points are drawn
%R = 1;                            % Unit radius

x1 = R*cos(angle);  y1 = R*sin(angle);               % Set a point at angle
x2 = R*cos(angle+2*pi/n);  y2 = R*sin(angle+2*pi/n); % Set the next point

hold on;
plot(C(2), C(1), 'g.');
h = line( C(2)+[x1 x2], C(1)+[y1 y2]);                              % Draw the lines

set(h, 'Color', 'g');

hold off;
