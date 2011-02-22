function [Mixture] = MatchingPursuitGaussianApproximation_WithCleanUp(Image, Sigmas, Kernels, G, L2_N, NbGaussians)
%
%
%
% [X, Y] = meshgrid(1:size(Image, 1), 1:size(Image, 1));
% X = X'; Y = Y';
% X = X(:);
% Y = Y(:);
% Points = [X, Y];

It = Image;
It_convs = zeros(size(Image, 1), size(Image, 2), length(Sigmas));


Mixture.Sigmas  = [];
Mixture.Mu      = [];
Mixture.Weights = [];


% some precomputations

% % precompute the gaussian convolution kernel
% Kernels = cell(length(Sigmas), 1);
% for i = 1:length(Sigmas)
%     windowsize =  2*(2*Sigmas(i))+1;
%     Kernels{i} = fspecial('Gaussian', windowsize, Sigmas(i));
% end
% 
% % precompute placed kernels and their L2 norm
% G = cell(size(Image, 1), size(Image, 2), length(Sigmas));
% L2_Norms = zeros(size(Image, 1), size(Image, 2), length(Sigmas));
% for i=1:length(Sigmas)
%     for ii = 1:size(Points, 1)
%         x = X(ii);
%         y = Y(ii);
%         G{x, y, i} = gaussianKernel(Image, [x, y], Sigmas(i), Kernels{i});
%         L2_Norms(x, y, i) = sqrt(sum(sum(G{x, y, i}.*G{x, y, i})));
%     end
% end
L2_Norms = sqrt(L2_N);

for i = 1:NbGaussians
    %progressbar(i, NbGaussians);
    for j = 1:length(Sigmas)
        It_convs(:,:, j)  = imfilter(It, Kernels{j})./L2_Norms(:,:,j);
    end
    
    argmax = find(abs(It_convs) == max(abs(It_convs(:))) );
    
    [x, y, r] = ind2sub(size(It_convs),  argmax(1));
    
    Mixture.Sigmas = cat(1, Mixture.Sigmas, Sigmas(r));
    Mixture.Mu     = cat(2, Mixture.Mu    , [x; y]);
    
    weight = It_convs(x, y, r) / L2_Norms(x, y, r);
    
    Mixture.Weights = cat(1, Mixture.Weights, weight);
    
    It = It - weight*G{x, y, r};
    %ReconstructedImage = ReconstructedImage + weight*G{x, y, r};
    
    %ResidualsToPlot(i) = sqrt( sum(( Image(:) - ReconstructedImage(:)).^2) );
end
%% cleanup
AA = cat(1, Mixture.Mu, Mixture.Sigmas');

[~, I] = unique(AA', 'rows', 'first');
Mu     = Mixture.Mu(:, I');
sigmas = Mixture.Sigmas(I);

I_convs = zeros(size(Image, 1), size(Image, 2), length(Sigmas));
for j = 1:length(Sigmas)
    I_convs(:,:, j) = imfilter(Image, Kernels{j});
end

Z = zeros(size(sigmas));

M = zeros(length(Z));

for i = 1:length(Z)
    Z(i) = I_convs(Mu(1, i), Mu(2, i), Sigmas == sigmas(i));
    for k = 1:length(Z)
        M(i, k) =  sum(sum(G{Mu(1, i), Mu(2, i), Sigmas == sigmas(i)}...
                         .*G{Mu(1, k), Mu(2, k), Sigmas == sigmas(k)} ) );
        M(k, i) = M(i, k);
    end
end


%keyboard;

Mixture.Mu = Mu;
Mixture.Sigmas = sigmas;
Mixture.Weights = M \ Z;
% residualGlobalReconstruction = EnergyImage - Z'*(M\Z);
% secondReconstruction = zeros(size(Image));
% for i = 1:length(Mixture.Weights)
%     
%     secondReconstruction = secondReconstruction + Mixture.Weights(i)*G{Mixture.Mu(1, i), Mixture.Mu(2, i), Sigmas == Mixture.Sigmas(i)};
%     
% end


