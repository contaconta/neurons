function r = randint(a, n)

if nargin < 2
    n = 1;
end

if isscalar(a)
    max = a;
    min = 1;
else
    max = a(2)+1;
    min = a(1);
end
r = floor( (max-min)*rand(1,n)) + min; 