function scale_estimate = scaleEstimate2(I)

MAX_REDUCTIONS = 2;
S = 3;
sigma0 = 1;  %.5
PYRAMID = 1;


p = [ -6.4925e-34  1.0796e-26 -5.6185e-20 1.1418e-13 -7.9515e-08 .2201];


% create a list of gaussian sigmas corresponding to various scales
%max_sigma = 2*min(size(I));
max_sigma = min(size(I));
%max_sigma = max(size(I));
%max_sigma = min(size(I)/6);
k = 2^(1/S);
sigma_list  = get_sigmas(sigma0, max_sigma,k);

G = zeros(size(I,1), size(I,2), length(sigma_list));
DoG = zeros(size(I,1), size(I,2), length(sigma_list)-1 );

I = double(I); I = impyramid(I, 'expand');


blah_vals = numel(I)./(sigma_list.^4);
polyvals = zeros(size(blah_vals));

for i = 1:length(blah_vals)
    polyvals(i) = polyval(p, blah_vals(i));
end

%keyboard;
%norm_consts = min(polyvals.^8) ./ (polyvals.^8);
%norm_consts = min(polyvals.^12) ./ (polyvals.^12);
%norm_consts = 1./(sigma_list);
norm_consts = 1./(sigma_list.^(.6));
%norm_consts = 1./(sigma_list - [0 sigma_list(1:length(sigma_list)-1)])

%norm_consts = 1  ./  (1:15/(length(sigma_list)-1):16);
%norm_consts = ones(size(sigma_list));



% norm_consts = [4.0125 3.2623 3.5440 3.6385 3.6708 3.3944 3.4357 2.9375 2.5501 2.3233 1.9544 1.0837 0.8066 0.5144 0.3455 0.2637 0.2264 0.2869 0.2258 0.3045 0.1855 0.1632 0.1355 .10]; 
% norm_consts = sqrt(norm_consts);
% norm_consts = norm_consts ./ sum(norm_consts);
%norm_consts = 1./norm_consts;    

tic;
s = 1;
for sigma_i = sigma_list
    %disp(['sigma_i = ' num2str(sigma_i)]);
    
    if PYRAMID
        sigma_red = sigma_i;  red_fact = 0;
        
        while ( (sigma_red / 2) >= 1 ) && (red_fact <= MAX_REDUCTIONS)
        %while (sigma_red / 2) >= 1
            red_fact = red_fact + 1;
            sigma_red = sigma_red/2;
        end

        Ired = I;
        for i = 1:red_fact
            %Ired = impyramid(Ired, 'reduce');
            Ired = imresize(Ired, .5);
        end
        disp([ 'size = [' num2str(size(Ired)) ']  kernel size = ' num2str(round(6*sigma_red + 1)) ' sigma_i = ' num2str(sigma_i/2), ' reduction factor = ' num2str(red_fact)  ]);
    end
    
    
    
 
    norm_const = norm_consts(s);

    % apply the gaussian
    if PYRAMID
        Gred = imgaussian(Ired, sigma_red);
        G(:,:,s) = imresize(Gred, [size(G,1) size(G,2)]);
    else
        disp(['sigma_i = ' num2str(sigma_i/2) ]);
        Gred = imgaussian(I, sigma_i);
        if ~isequal(size(Gred), [size(G,1) size(G,2)])
            G(:,:,s) = imresize(Gred, [size(G,1) size(G,2)]);
        else
            G(:,:,s) = Gred;
        end
    end

    % compute the DoG
    if s > 1
        DoG(:,:,s-1) = abs(G(:,:,s) - G(:,:,s-1))*norm_const;
        %figure; imagesc( DoG(:,:,s-1) ); axis image; colormap gray;
    end
    
    s = s + 1;
end
toc;

% reverse the order of DoG and sigma_list because max always picks the
% first in case of multiple maximal elements
DoG = DoG(:,:, size(DoG,3):-1:1);
sigma_list = .5*sigma_list(length(sigma_list):-1:1);
    
[M maxDoG] = max(DoG, [], 3);
scale_estimate = maxDoG;


a1 = squeeze(median(median(DoG,1),2));
a2 = squeeze(mean(mean(DoG,1),2));
a3 =squeeze(max(max(DoG,[],1),[],2));
[a1 a2 a3]

% %absDoG = abs(DoG);
% %fun = @(x) localpeak(x);
% BW1 = imregionalmax(DoG, 26);
% BW2 = imregionalmin(DoG, 26);
% %DoGpeaks = nlfilter(absDoG, [3 3 3], fun);
% DoGPeaks = (BW1 | BW2) .* DoG;
% [M maxPeaks] = max(DoGPeaks, [], 3);
% 
% for s = 1:length(sigma_list)
%   scale_estimate(maxDoG == s) = sigma_list(s);
% end


figure; imagesc(abs(s - maxDoG)); axis image; colormap gray;
%figure; imagesc(scale_estimate); axis image; colormap gray;

filtsize = round([  max(size(I))/300  max(size(I))/300])

maxDoG2 = medfilt2(maxDoG, filtsize);
figure; imagesc(abs(s - maxDoG2)); axis image; colormap gray;

% A = zeros([size(DoG,1) size(DoG,2)]);
% for r = 1:size(DoG,1)
%     for c = 1:size(DoG,2)
%         a = squeeze(DoG(r,c,:));
%         [pks, locs] = findpeaks( squeeze(DoG(r,c,:)));
%         A(r,c) = locs(length(locs));
%     end
% end

keyboard;

% 
% function val = localpeak(A)
% midpt = round(size(A,1)/2);
% x = A(midpt,midpt,2);    % assuming A is NxNx3 
% if x >= max(A(:))
%     val = x;
% else
%     val = 0;
% end

function sigma_list  = get_sigmas(sigma_list, max_sigma,k)

if sigma_list(length(sigma_list))*k < max_sigma
    sigma_list = [sigma_list sigma_list(length(sigma_list))*k];
    %keyboard;
    sigma_list = get_sigmas(sigma_list, max_sigma, k);
end


