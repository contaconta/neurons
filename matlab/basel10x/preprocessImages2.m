function [log1, f, J] = preprocessImages2(R, G, LoG, opt)

log1 = cell(size(R));
f = cell(size(R));
J = cell(size(R));

parfor  t = 1:length(R)
    Rt = mat2gray(double(R{t}));
    log1{t} = WLV(LoG, Rt, 1e-4);
    J{t} = mat2gray(double(G{t}));
    f{t} = FrangiFilter2D(J{t}, opt);
end
