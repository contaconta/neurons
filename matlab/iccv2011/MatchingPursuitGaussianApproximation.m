function [X Y W S Mixture]= MatchingPursuitGaussianApproximation(Image, Sigmas, NbGaussians)
% 
%   
%
addpath('anigaussm/');

IMSIZE = size(Image);
PAD = round(  size(Image)/3 );
Image = padarray(Image, PAD);

[X, Y] = meshgrid(1:size(Image, 1), 1:size(Image, 1));
X = X'; Y = Y';

It = Image;
It_convs = zeros(size(Image, 1), size(Image, 2), length(Sigmas));

%ReconstructedImage = zeros(size(Image));

Mixture.Sigmas  = [];
Mixture.Mu      = [];
Mixture.Weights = [];

for i = 1:NbGaussians
   for j = 1:length(Sigmas)
       It_convs(:,:, j) = Sigmas(j)*anigauss_mex(It, Sigmas(j));
   end
   
   argmax = find(abs(It_convs) == max(abs(It_convs(:))) );
   
   [x, y, r] = ind2sub(size(It_convs),  argmax(1));
   
   %keyboard;
   
   Mixture.Sigmas = cat(1, Mixture.Sigmas, Sigmas(r));
   Mixture.Mu     = cat(2, Mixture.Mu    , [x; y]);
   

   normalizationWeight = 1/ (4*pi*Sigmas(r));
   weight = It_convs(x, y, r) / normalizationWeight;
   
   Mixture.Weights = cat(1, Mixture.Weights, weight);
   
   It = It - weight*gaussian(X, Y, [x, y], Sigmas(r));
   %ReconstructedImage = ReconstructedImage + weight*gaussian(X, Y, [x, y], Sigmas(r));
end


%%====================== KEVINS STUFF ====================================

% put the GMM into a format I like
X = Mixture.Mu(2,:)-PAD(2);
X = X(:);
Y = Mixture.Mu(1,:)-PAD(1);
Y = Y(:);
W = Mixture.Weights;
S = Mixture.Sigmas;

% remove any gaussians placed outside of the image due to padding
badinds1 = (X < 1);
badinds2 = (Y < 1);
badinds3 = (X > IMSIZE(2));
badinds4 = (Y > IMSIZE(1));

inds = ~(badinds1 | badinds2 | badinds3 | badinds4);
X = X(inds);
Y = Y(inds);
W = W(inds);
S = S(inds);

% mean center according to area (assume 2*sigma radius)
winds = (W > 0);
binds = (W <= 0);
warea = sum( 4*pi*S(winds).^2);
barea = sum( 4*pi*S(binds).^2);
W(winds) = W(winds)/warea;
W(binds) = W(binds)/barea;
