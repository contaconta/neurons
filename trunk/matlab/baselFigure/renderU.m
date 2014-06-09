function mv = renderU(U, Green, NeuriteDetectionParams)

%GEODESIC_DISTANCE_NEURITE_THRESH = NeuriteDetectionParams.GEODESIC_DISTANCE_NEURITE_THRESH;
ProbThresh                       = NeuriteDetectionParams.NeuriteProbabilityThreshold;

% figure; imagesc(exp(-U{1}));
% 
% 
% figure; imagesc(  exp(-U{1}) > ProbThresh);


% h1 = figure; 
% h2 = figure;
% h3 = figure;


TMAX = length(U);
for t = 1:TMAX
    
    
    prob = exp(-U{t});
    
%     figure(h1);
%     imagesc(prob);
    
    
%     figure(h2);
    mask = prob > ProbThresh;
    inds = find(mask);
%     imagesc(mask);
    
    
%     figure(h3);
    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    prob = prob / max(prob(:));
    %Ir(mask) = 0; %prob(mask);
    %Ig(mask) = 0; % prob(mask);
    
    for i = 1:numel(inds)
        p = prob(inds(i));
        [r, g, b] = colorlookup(p);
        Ir(inds(i)) = r;
        Ig(inds(i)) = g;
        Ib(inds(i)) = b;
    end
    
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    mv{t} = I;
    
%     imshow(I);
%     refresh;
%     drawnow;
%     pause(.1);
    
end





function [r,g,b] = colorlookup(p)
% 
% r = .25*p + .5*sin(p);
% g = .25*p + .5*sin(p);
% b = .25+ cos(p);

r = (p-.1)^1.8 -.1; %.2*p + .35*sin(p);
g = (p-.1)^1.8 -.1; %.2*p + .35*sin(p);
b = p^(.8); %.4 + cos(p);

