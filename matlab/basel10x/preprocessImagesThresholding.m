function [log1, f, J] = preprocessImagesThresholding(R, G, LoG, opt)

log1 = cell(size(R));
f = cell(size(R));
J = cell(size(R));
parfor  t = 1:length(R)
    Rt = mat2gray(double(R{t}));
    J{t} = mat2gray(double(G{t}));
    log1{t} = imfilter(Rt, LoG);
    f{t} = FrangiFilter2D(J{t}, opt);
end
