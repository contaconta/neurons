function [Mixture residualGlobalReconstruction] = MatchingPursuitGaussianApproximationFindNbGaussians(Image, Sigmas, Kernels, G, L2_N, Tolerance, MaxNbGaussians)
%
%
%

It = Image;
It_convs = zeros(size(Image, 1), size(Image, 2), length(Sigmas));


Mixture.Sigmas  = [];
Mixture.Mu      = [];
Mixture.Weights = [];

EnergyImage = sum(Image(:).^2);

L2_Norms = sqrt(L2_N);
residualGlobalReconstruction = 1e9;
i = 0;
while i < MaxNbGaussians &&  residualGlobalReconstruction > Tolerance
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
    AA = cat(1, Mixture.Mu, Mixture.Sigmas');
    
    [tmp, I] = unique(AA', 'rows', 'first'); clear tmp;
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
    
    
    
    Mixture.Mu = Mu;
    Mixture.Sigmas = sigmas;
    Mixture.Weights = M \ Z;
    residualGlobalReconstruction = (EnergyImage - Z'*(M\Z))/numel(Image);
    
    
    i = i+1;
end


