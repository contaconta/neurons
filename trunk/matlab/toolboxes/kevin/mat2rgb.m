function rgb = mat2rgb(X, map)

if nargin == 1
    map = jet;
end

g = mat2gray(X);
x = gray2ind(g, 256);
rgb = ind2rgb(x, jet(256));



