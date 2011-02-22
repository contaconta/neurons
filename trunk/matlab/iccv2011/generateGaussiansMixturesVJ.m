Sigmas = 8:-1:1;
Sigmas(end+1) = 0.5;


% some precomputations

% precompute the gaussian convolution kernel
Kernels = cell(length(Sigmas), 1);
for i = 1:length(Sigmas)
    windowsize =  2*(2*Sigmas(i))+1;
    Kernels{i} = fspecial('Gaussian', windowsize, Sigmas(i));
end

% precompute placed kernels and their L2 norm
Image = zeros(IMSIZE + 2*PAD);

[X, Y] = meshgrid(1:size(Image, 1), 1:size(Image, 1));
X = X'; Y = Y';
X = X(:);
Y = Y(:);
Points = [X, Y];


G = cell(size(Image, 1), size(Image, 2), length(Sigmas));
L2_Norms = zeros(size(Image, 1), size(Image, 2), length(Sigmas));
for i=1:length(Sigmas)
    for ii = 1:size(Points, 1)
        x = X(ii);
        y = Y(ii);
        G{x, y, i} = gaussianKernel(Image, [x, y], Sigmas(i), Kernels{i});
        L2_Norms(x, y, i) = sum(sum(G{x, y, i}.*G{x, y, i}));
    end
end

