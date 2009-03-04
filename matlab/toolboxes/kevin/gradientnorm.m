
function v = gradientnorm(v)

mag = sqrt(  sum( (v.^2),3));
v = v ./ repmat(mag, [1 1 2]);


%v = v / sqrt( sum(sum(sum(v.^2))) + eta2);