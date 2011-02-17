function G = gaussian(X, Y, center,r)

 
if r > 0
    G = 1/ (2*pi*r*r)*exp( -((X-center(1)).^2 + (Y-center(2)).^2) / (2*r*r) );
else
    G = zeros(size(X));
    G(center(2),center(1)) = 1/(2*pi);
end