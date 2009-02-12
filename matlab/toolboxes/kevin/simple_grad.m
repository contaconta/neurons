function [NORM, ANGL] = simple_grad(I)



filtX = [1 0 -1];
filtY = filtX';

GradX = imfilter(I,filtX, 'symmetric');
GradY = imfilter(I,filtY, 'symmetric');


% compute the gradient magnitude, and gradient orientation
NORM = arrayfun(@(a,b)(norm([a b])), GradX, GradY);
ANGL = arrayfun(@(a,b)(atan2(a,b)), GradY, GradX);