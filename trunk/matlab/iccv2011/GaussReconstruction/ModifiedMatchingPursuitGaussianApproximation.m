function [ Mixture] = ModifiedMatchingPursuitGaussianApproximation(Image, Sigmas, Kernels, G, L2_Norms, NbGaussians)
%
%
%

[X, Y] = meshgrid(1:size(Image, 1), 1:size(Image, 1));
X = X'; Y = Y';
X = X(:);
Y = Y(:);
Points = [X, Y];

Residuals = zeros(size(Image, 1), size(Image, 2), length(Sigmas));

%ResidualsToPlots = zeros(NbGaussians, 1);

% some precomputations

% % precompute the gaussian convolution kernel
% Kernels = cell(length(Sigmas), 1);
% for i = 1:length(Sigmas)
%     windowsize =  2*(2*Sigmas(i))+1;
%     Kernels{i} = fspecial('Gaussian', windowsize, Sigmas(i));
% end

% precompute convolutions with gaussians 
I_convs = zeros(size(Image, 1), size(Image, 2), length(Sigmas));
for j = 1:length(Sigmas)
    I_convs(:,:, j) = imfilter(Image, Kernels{j});
end

% % precompute placed kernels and their L2 norm
% G = cell(size(Image, 1), size(Image, 2), length(Sigmas));
% L2_Norms = zeros(size(Image, 1), size(Image, 2), length(Sigmas));
% for i=1:length(Sigmas)
%     for ii = 1:size(Points, 1)
%         x = X(ii);
%         y = Y(ii);
%         G{x, y, i} = gaussianKernel(Image, [x, y], Sigmas(i), Kernels{i});
%         L2_Norms(x, y, i) = sum(sum(G{x, y, i}.*G{x, y, i}));
%     end
% end

% initialize the mixture
Mixture.Sigmas  = [];
Mixture.Mu      = [];
Mixture.Weights = [];

% main for loop on the number of desired gaussians
for i = 1:NbGaussians
    %progressbar(i, NbGaussians);
    
    Z = zeros(length(Mixture.Sigmas) + 1, 1);
    M = zeros(length(Mixture.Sigmas) + 1);
    
    GG = cell(length(Mixture.Sigmas)+1, 1);
    for j=1:length(Mixture.Sigmas)
        Z(j) =  I_convs( Mixture.Mu(1, j), Mixture.Mu(2, j), Sigmas == Mixture.Sigmas(j) );
        Gj = G{Mixture.Mu(1, j), Mixture.Mu(2, j), Sigmas == Mixture.Sigmas(j)};
        GG{j} = Gj;
        for k = 1:length(Mixture.Sigmas)
            Gk = G{Mixture.Mu(1, k), Mixture.Mu(2, k), Sigmas == Mixture.Sigmas(k)};
            M(j, k) =  sum(Gj(:).*Gk(:));
            M(k, j) = M(j, k);
        end
    end
    % TODO : this for loop should be parfor
    for j=1:length(Sigmas)
        
        for k = 1:numel(Image)
            x = Points(k, 1);
            y = Points(k, 2);
            Z(end) = I_convs( x, y, j );
            GG{end}  = G{x, y, j};
            
            for l = 1:length(Mixture.Sigmas)
                M(end, l) = sum(sum(GG{end}.*GG{l}));
                M(l, end) = M(end, l);
            end
            
            M(end, end) = L2_Norms(x, y, j);
            
            if( i >1 )
                tab = find(x == Mixture.Mu(1, :) & y == Mixture.Mu(2, :) & Sigmas(j) == Mixture.Sigmas');
                if(numel(tab) == 1)
                    Residuals(x, y, j) = Z(1:end-1)'* (M(1:end-1, 1:end-1) \ Z(1:end-1));
                else
                    Residuals(x, y, j) = Z'* (M \ Z);
                end
                
            else
                Residuals(x, y, j) = Z'* (M \ Z);
            end
        end
    end
    argmax = find(Residuals == max(Residuals(:)) );
    % we got the best one
    [x, y, r] = ind2sub(size(Residuals),  argmax(1));
    
    Mixture.Mu     = cat(2, Mixture.Mu    , [x; y]);
    Mixture.Sigmas = cat(1, Mixture.Sigmas, Sigmas(r));
    
    Z(end) = I_convs( x, y, r);
    
    GG{end} = G{x, y, r};
    
    for k = 1:length(Z)
        M(end, k) =  sum(sum(GG{end}.*GG{k}));
        M(k, end) = M(end, k);
    end
    %keyboard;
    
    
    % reconstruct only if at the last point
    %ReconstructedImage = zeros(size(Image));
    Weights = M \ Z;
    Mixture.Weights = Weights;
%     for j = 1:length(Weights)
%         ReconstructedImage = ReconstructedImage + ...
%             Mixture.Weights(j)*GG{j};
%     end
    %ResidualsToPlots(i) = sqrt( sum(( Image(:) - ReconstructedImage(:)).^2) );
    
end

