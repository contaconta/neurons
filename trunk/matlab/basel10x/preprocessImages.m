function [log1, f, J, M] = preprocessImages(R, G, LoG, minArea, maxArea, opt)

log1 = cell(size(R));
f = cell(size(R));
J = cell(size(R));
M = cell(size(R));

parfor  t = 1:length(R)
    Rt = mat2gray(double(R{t}));
    
    % 	Ellipses{t} = vl_ertr(Ellipses{t});
    J{t} = mat2gray(double(G{t}));
    log1{t} = imfilter(Rt, LoG);
%     I = log1{t};
%     I = uint8(255*(I-min(I(:)))/(max(I(:)) - min(I(:))));
%     log1{t} = I;
%     r = vl_mser(I, 'MinDiversity', 0.2,...
%         'MaxVariation', 0.25,...
%         'MinArea', minArea/numel(I), ...
%         'MaxArea', maxArea/numel(I), ...
%         'BrightOnDark', 1, ...
%         'Delta',1) ;
%     
%     
%     M{t} = r;
    
    f{t} = FrangiFilter2D(J{t}, opt);
end
