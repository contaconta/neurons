function Pr = oof(imageName)

Im = double(imread(imageName));
Im = sum(Im, 3);
Im = rescale(Im, 0, 1);
%%
R = [1:10];
h = [1;1];
%%
tic
[TT] = ScaledHessianGaussian2D([1;1], Im, R);
toc
%%
Pr = squeeze(TT(:,:, 1, :) + TT(:,:, 2, :));
Pmax = max(Pr,[], 3);

