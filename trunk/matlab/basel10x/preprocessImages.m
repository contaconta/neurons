function [log1, f, J] = preprocessImages(R, G, LoG, opt)

log1 = cell(size(R));
f = cell(size(R));
J = cell(size(R));

parfor  t = 1:length(R)
    Rt = mat2gray(double(R{t}));
    log1{t} = imfilter(Rt, LoG, 'replicate');
    J{t} = mat2gray(double(G{t}));
    f{t} = FrangiFilter2D(J{t}, opt);
end
