function [Tubularity, J] = trkComputeTubularity(Green, opt)

TMAX = length(Green);
pad = 3*opt.FrangiScaleRange(2);

Tubularity = cell(size(Green));
J          = cell(size(Green));

parfor t = 1:TMAX
    J{t}          = mat2gray(Green{t});
    Tubularity{t} = FrangiFilter2D(J{t}, opt);
    Mask = ones(size(Tubularity{t}));
    Mask(pad:end-pad, pad:end-pad) = 0;
    Tubularity{t}(Mask > 0.5) = min(Tubularity{t}(Mask <= 0.5));
end