function  EDGE = zerocrossing(I, sigma, thresh)


% N = ceil(sigma*3)*2+1;
% HSIZE = [N N];
% 
% 
% h = fspecial('log',HSIZE,sigma);
% F = 4*imfilter(I, h, 'symmetric') + .5;


EDGE = edge(I, 'log', 0, sigma);

[NORM, ANGL] = simple_grad(I);

EDGE = EDGE & (NORM > thresh);


