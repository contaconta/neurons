function SCALESPACE = DoG(V, sigma, k)
%V = DoG(V, sigma1, k)
%
%Applies a difference of Gaussian (DoG) filter to n-dimensional data
%V.  Sigma1 is a vector with the standard devaitions of the outer 
%Gaussian (i.e. [1.6 1.6] for a symmetric 2-D DoG filter, and
%the inner std. dev. is given by sigma2 = k*sigma1, where k < 0.  The
%default value for k is 1.2.

%scales = [1 1/2 1/4 1/8 ];
scales = 1;

if nargin == 2
    k = 2^(1/3); % 1.2;
end

% STEP 1: CHECK THE INPUTS
if size(V,3) <= 3
    dims = 2;
else
    dims = 3;
end

if dims > (length(sigma))
    sigma = sigma*(ones(length(size(V)),1))';
end



disp('...applying difference of Gaussian (DoG) filter...');
SCALESPACE = zeros(size(V));
% FOR EACH SCALE FACTOR IN THE SCALE SPACE
for s = 1:length(scales)
    
    % Scale the volume
    clear imS; clear f;
%     t = maketform('affine', [scales(s) 0 0; 0 scales(s) 0; 0 0 1]);
%     r = makeresampler('nearest', 'fill');
%     VS = tformarray(V,t,r,[1 2],[1 2], [round(scales(s)*size(V,1)) round(scales(s)*size(V,2))], [],[]);
 
VS = V;

    % STEP 2: DECIDE IF DATA IS 2D or 3D
    %==================== 2D CASE ===================
    if dims == 2
        if size(V,3) ~= 1
            VS = rgb2gray(VS);
        end
        dx1 = gauss0(sigma(1));
        dy1 = gauss0(sigma(2))';
        VS1 = imfilter(VS, dx1, 'conv', 'replicate');
        VS1 = imfilter(VS1, dy1, 'conv', 'replicate');
        dx2 = gauss0(k*sigma(1));
        dy2 = gauss0(k*sigma(2))';
        VS2 = imfilter(VS, dx2, 'conv', 'replicate');
        VS2 = imfilter(VS2, dy2, 'conv', 'replicate');
        VS = VS1 - VS2; %VS2 - VS1;
            
    %==================== 3D CASE ===================    
    else
        V1 = zeros(size(V)); V2 = V1; 
        for z = 1:size(V,3);
            XYslice = squeeze(double(V(:,:,z)));
            dx1 = gauss1(sigma(1));
            Ix1 = imfilter(XYslice, dx1,'conv', 'replicate');
            V1(:,:,z) = Ix1;
            dx2 = gauss1(k*sigma(1));
            Ix2 = imfilter(XYslice, dx2,'conv', 'replicate');
            V2(:,:,z) = Ix2;
        end
        for z = 1:size(V,3);
            XYslice = squeeze(double(V(:,:,z)));
            dy1 = gauss0(sigma(2))';
            Iy1 = imfilter(XYslice, dy1,'conv', 'replicate');
            V1(:,:,z) = Iy1;
            dy2 = gauss0(k*sigma(2))';
            Iy2 = imfilter(XYslice, dy2,'conv', 'replicate');
            V2(:,:,z) = Iy2;
        end
        for y = 1:size(V,2);
            XZslice = squeeze(double(V(:,y,:)));
            dz1 = gauss0(sigma(3))';
            Iz1 = imfilter(XZslice, dz1,'conv', 'replicate');
            V1(:,y,:) = Iz1;
            dz2 = gauss0(k*sigma(3))';
            Iz2 = imfilter(XZslice, dz2,'conv', 'replicate');
            V2(:,y,:) = Iz2;
        end
        VS = V1 - V2;    

    end

    %%%%% SCALE FEATURES UP %%%%%%%%
%     t = maketform('affine', [1/scales(s) 0 0; 0 1/scales(s) 0; 0 0 1]);  
%     r = makeresampler('nearest', 'fill');
%     VS = tformarray(VS,t,r,[1 2],[1 2], [size(V,1) size(V,2)], [],[]);
%     
%     SCALESPACE = max(SCALESPACE, VS);

SCALESPACE = VS;
end


function g0 = gauss0(sigma)
%G = gauus0(sigma)
%
%Gives a 1-D approximate gaussian distribution, G, with an 
%appropriately sized window length for its standard deviation, sigma.

g0 = normpdf(-100:100, 0, sigma);

g0 = g0(find(g0 > .0000001 * max(g0)))';



% 
% 
% %==================== OUTER FILTER DEFINITION ====================
% function [g0, inds] = Gout(sigma)
% 
% g0 = normpdf(-100:100, 0, sigma);
% inds = find(g0 > .0000001 * max(g0));
% g0 = g0(inds);
% 
% 
% %==================== INNER FILTER DEFINITION ====================
% function g0 = Gin(sigma, inds)
% 
% g0 = normpdf(-100:100, 0, sigma);
% g0 = g0(inds);
% 
% 
% %==================== COMBINED FILTER DEFINITION ====================
% function f = Gfilt(sigma,k)
% 
% [gout, inds] = Gout(k*sigma);
% gin = Gin(sigma, inds);
% f = gout- gin;
% 
